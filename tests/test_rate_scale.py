"""The learning-rate multiplier: warm-up, constant, then optional token-keyed decay."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch" / "dense_gr"))


def test_rate_scale_warms_holds_and_decays_to_its_floor():
    from smoke_train import rate_scale

    assert rate_scale(0, 100, 0.0) == 0.01
    assert rate_scale(99, 100, 0.1) == 1.0
    # No decay requested: constant for the rest of the run, as every earlier run was.
    assert rate_scale(500, 100, 0.99) == 1.0
    # Decay over the last 20% of tokens: untouched before it, linear through it.
    assert rate_scale(500, 100, 0.79, 0.2) == 1.0
    assert rate_scale(500, 100, 0.9, 0.2) == 1.0 - 0.9 * 0.5
    assert abs(rate_scale(500, 100, 1.0, 0.2, 0.1) - 0.1) < 1e-12
    # A run that overshoots its budget stays at the floor rather than going negative.
    assert abs(rate_scale(500, 100, 1.2, 0.2, 0.1) - 0.1) < 1e-12