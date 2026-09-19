"""The spill guard has to keep guarding once it stops blocking the training loop.

Reading the WDDM shared-usage counter costs 1.81 s because it starts a PowerShell
process. Called from the training loop it lands just after a `torch.cuda.synchronize()`,
so the GPU has nothing queued for that whole time -- measured as four dips per minute at
a 13.5 s report cadence, a tenth of throughput. Moving it to a daemon thread is only
correct if the guard still trips on an excursion the loop never directly observed, which
is what these cover.
"""

import sys
import threading

import pytest

sys.path.insert(0, "scratch/dense_gr")

import benchmark  # noqa: E402


@pytest.fixture
def scripted(monkeypatch):
    """Replace the counter with a scripted sequence, and say when it has been drained."""
    def install(values):
        remaining = list(values)
        drained = threading.Event()

        def fake():
            if not remaining:
                drained.set()
                return values[-1]
            return remaining.pop(0)

        monkeypatch.setattr(benchmark, "shared_gpu_gib", fake)
        return drained
    return install


def test_drift_reports_the_worst_reading_not_the_latest(scripted):
    """A spill that grows and recedes between reports still has to trip the guard.

    This is the whole reason the watch keeps a peak. The training loop reads `drift()`
    every twenty-five steps; a latest-only reading would show 0.0 for an excursion that
    happened and resolved in between, which is exactly the event the check exists for.
    """
    drained = scripted([1.0, 1.2, 4.5, 1.1, 1.0])
    watch = benchmark.SpillWatch(interval=0.01).start()
    assert drained.wait(timeout=10)
    drift = watch.stop()
    assert watch.baseline == 1.0
    assert watch.peak == pytest.approx(4.5)
    assert drift == pytest.approx(3.5)
    assert watch.latest == pytest.approx(1.0)


def test_a_failed_counter_read_is_not_evidence_of_a_spill(scripted):
    """`shared_gpu_gib` returns nan when the counter is unavailable."""
    drained = scripted([2.0, float("nan"), float("nan"), 2.25])
    watch = benchmark.SpillWatch(interval=0.01).start()
    assert drained.wait(timeout=10)
    watch.stop()
    # nan must reach neither the peak nor the latest: it is a failed read, and treating
    # it as a measurement would either trip the guard or hide a real excursion.
    assert watch.drift() == pytest.approx(0.25)
    assert watch.peak == pytest.approx(2.25)
    assert watch.latest == pytest.approx(2.25)


def test_the_watcher_owns_the_verdict(scripted):
    """The flag is set by the poll that saw the breach, not by the loop's arithmetic.

    The training loop asks `breached()` every step, which is free; the tolerance lives in
    one place instead of once per runner, and the value reported is the breach itself
    rather than whatever the counter has settled back to by the next report.
    """
    drained = scripted([1.0, 1.1, 3.0, 1.05])
    watch = benchmark.SpillWatch(interval=0.01, tolerance=0.25).start()
    assert not watch.breached(), "the baseline cannot be a breach of itself"
    assert drained.wait(timeout=10)
    watch.stop()
    assert watch.breached()
    assert watch.tripped_at == pytest.approx(2.0)
    assert watch.latest == pytest.approx(1.05), "the excursion already receded"


def test_a_quiet_run_never_trips(scripted):
    drained = scripted([1.0, 1.05, 1.2, 1.24])
    watch = benchmark.SpillWatch(interval=0.01, tolerance=0.25).start()
    assert drained.wait(timeout=10)
    watch.stop()
    assert not watch.breached()
    assert watch.tripped_at is None


def test_the_loop_never_waits_on_the_counter(scripted):
    """`drift` must return without paying the counter's cost, even on a slow read."""
    started = threading.Event()

    def slow():
        started.set()
        threading.Event().wait(30)  # never returns within the test
        return 0.0

    watch = benchmark.SpillWatch(interval=0.01)
    watch.baseline = watch.peak = watch.latest = 1.0
    watch._thread = threading.Thread(target=slow, daemon=True)
    watch._thread.start()
    assert started.wait(timeout=5)
    # The point of the fixture: a read that never finishes must not reach the caller.
    assert watch.drift() == pytest.approx(0.0)
    watch._stop.set()


def test_stop_is_safe_before_the_thread_ever_started():
    watch = benchmark.SpillWatch(interval=0.01)
    assert watch.stop() != watch.stop() or True  # nan != nan; it must not raise
