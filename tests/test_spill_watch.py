"""The spill guard has to keep guarding once it stops blocking the training loop.

Reading the WDDM shared-usage counter costs 1.81 s because it starts a PowerShell
process. Called from the training loop it lands just after a `torch.cuda.synchronize()`,
so the GPU has nothing queued for that whole time -- measured as four dips per minute at
a 13.5 s report cadence, a tenth of throughput. Moving it to a daemon thread is only
correct if the guard still trips on an excursion the loop never directly observed, which
is what these cover.
"""

import json
import math
import subprocess
import sys
import threading
from types import SimpleNamespace

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
    assert math.isnan(watch.stop())
    assert math.isnan(watch.stop())
    assert watch.attempts == 0
    assert watch.report()["spilled"] is None


def test_late_baseline_recovers_guard_but_not_missing_history(scripted):
    scripted([float("nan"), 2.0, 3.5])
    watch = benchmark.SpillWatch(tolerance=1.0)
    watch._sample()
    assert watch.report()["spill_telemetry"]["coverage"] == "unavailable"
    with pytest.warns(RuntimeWarning, match="baseline established late"):
        watch._sample()
    assert watch.baseline == 2.0
    assert watch.report()["spilled"] is None
    watch._sample()
    assert watch.breached()
    report = watch.report()
    assert report["spilled"] is True
    assert report["spill_telemetry_valid"] is False
    assert report["spill_telemetry"]["baseline_delayed"] is True
    assert report["shared_delta_gib"] == 1.5
    json.dumps(report, allow_nan=False)


def test_failed_midrun_sample_preserves_peak_but_invalidates_clean_verdict(scripted):
    scripted([1.0, float("inf"), 1.1])
    watch = benchmark.SpillWatch()
    for _ in range(3):
        watch._sample()
    assert watch.peak == 1.1
    assert not watch.breached()
    assert watch.report()["spilled"] is None
    assert watch.report()["spill_telemetry"]["failed_reads"] == 1


def test_short_run_gets_final_sample_and_uses_watchers_threshold(scripted):
    scripted([1.0, 1.5])
    watch = benchmark.SpillWatch(interval=3600, tolerance=1.0).start()
    assert watch.report()["spilled"] is None  # baseline alone is not coverage
    assert watch.stop() == .5
    assert watch.report()["spilled"] is False  # not the old hard-coded .25
    assert watch.report()["spill_telemetry_valid"] is True
    assert watch.polls == 2
    watch.stop()
    assert watch.polls == 2  # idempotent


def test_final_sample_can_trip_guard(scripted):
    scripted([1.0, 2.5])
    watch = benchmark.SpillWatch(interval=3600, tolerance=1.0).start()
    watch.stop()
    assert watch.report()["spilled"] is True


def test_reader_pins_native_module_and_exact_pid(monkeypatch):
    seen = {}

    def run(argv, **kwargs):
        seen.update(argv=argv, kwargs=kwargs)
        return SimpleNamespace(returncode=0, stdout='{"scope":"process","samples":2,"bytes":2147483648}', stderr='')

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    monkeypatch.setattr(benchmark.os, "getpid", lambda: 12)
    assert benchmark.shared_gpu_gib() == 2.0
    assert benchmark.shared_gpu_gib.scope == "process"
    script = seen["argv"][-1]
    assert "Join-Path $PSHOME 'Modules\\Microsoft.PowerShell.Diagnostics" in script
    assert "Memory(pid_12_*)" in script
    assert "GPU Adapter Memory" not in script
    assert "Invalid counter status" in script
    assert "-NonInteractive" in seen["argv"]
    assert seen["kwargs"]["timeout"] == 10


@pytest.mark.parametrize("stdout", [
    "adapter ", "{}", "null", "[]",
    '{"scope":"adapter","samples":1,"bytes":0}',
    '{"scope":"process","samples":0,"bytes":0}',
    '{"scope":"process","samples":1,"bytes":-1}',
    '{"scope":"process","samples":1,"bytes":NaN}',
    '{"scope":"process","samples":1,"bytes":Infinity}',
    '{"scope":"process","samples":1,"bytes":true}',
])
def test_invalid_reader_output_is_unknown_and_visible(monkeypatch, stdout):
    monkeypatch.setattr(benchmark.shared_gpu_gib, "last_error", None, raising=False)
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout=stdout, stderr=''))
    with pytest.warns(RuntimeWarning, match="telemetry unavailable"):
        assert math.isnan(benchmark.shared_gpu_gib())
    assert benchmark.shared_gpu_gib.scope is None
    assert benchmark.shared_gpu_gib.last_error


def test_subprocess_error_exposed_then_cleared_on_success(monkeypatch):
    monkeypatch.setattr(benchmark.shared_gpu_gib, "last_error", None, raising=False)
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=1, stdout='', stderr='module import failed'))
    with pytest.warns(RuntimeWarning, match="module import failed"):
        assert math.isnan(benchmark.shared_gpu_gib())
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout='{"scope":"process","samples":1,"bytes":0}', stderr=''))
    assert benchmark.shared_gpu_gib() == 0
    assert benchmark.shared_gpu_gib.last_error is None


def test_subprocess_timeout_is_unknown(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("powershell", 10)

    monkeypatch.setattr(benchmark.shared_gpu_gib, "last_error", None, raising=False)
    monkeypatch.setattr(benchmark.subprocess, "run", timeout)
    with pytest.warns(RuntimeWarning, match="TimeoutExpired"):
        assert math.isnan(benchmark.shared_gpu_gib())
