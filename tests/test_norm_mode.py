"""The cheaper norms are available, and off wherever a donor has to be reproduced.

`x.to(fp32).pow(2).mean(-1)` writes two fp32 copies of the stream to produce one
reduction. `vector_norm` writes neither and is about eight times faster, at the cost of
roughly 1e-7 relative -- four orders below bf16's own resolution, and still fatal to
`recipient_initialize`, which reproduces a pretrained sublayer *bitwise* through this
route. So it is a per-module choice that defaults to exact.
"""

import sys

import pytest
import torch

sys.path.insert(0, "scratch/dense_gr")

from distillkit.experimental.hyper_connection import HyperConnection  # noqa: E402
from distillkit.experimental.widened_residual import _BranchNorm  # noqa: E402

HIDDEN, BRANCHES = 16, 4


def test_the_default_is_exact():
    route = HyperConnection(hidden_size=HIDDEN, num_branches=BRANCHES, lowrank=8)
    assert route.norm_mode == "exact"


def test_the_fast_path_agrees_to_far_better_than_bfloat16():
    torch.manual_seed(0)
    x = torch.randn(4, 32, BRANCHES, HIDDEN, dtype=torch.float32)
    gain = 1.0 + torch.randn(BRANCHES, HIDDEN, dtype=torch.float32) * 0.1
    exact = _BranchNorm.apply(x, gain, 1e-6, False)
    fast = _BranchNorm.apply(x, gain, 1e-6, True)
    relative = ((exact - fast).abs().max() / exact.abs().max()).item()
    assert relative < 1e-5, relative
    assert not torch.equal(exact, fast), "if these agree bitwise the test proves nothing"


def test_exact_and_fast_reach_the_module_as_asked():
    """A module built exact must stay bitwise exact; one built fast must not be."""
    torch.manual_seed(0)
    states = torch.randn(2, 8, BRANCHES, HIDDEN)
    built = {}
    for exact in (True, False):
        torch.manual_seed(1)
        route = HyperConnection(hidden_size=HIDDEN, num_branches=BRANCHES, lowrank=8,
                                blend=1.0, norm_mode="exact" if exact else "fast")
        with torch.no_grad():
            built[exact] = route._normalize(states)
    reference = _BranchNorm.apply(states, 1.0 + torch.zeros(BRANCHES, HIDDEN), 1e-6, False)
    assert torch.equal(built[True], reference)
    assert not torch.equal(built[False], reference)
    assert torch.allclose(built[False], reference, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("exact", [True, False])
def test_the_config_carries_the_choice_into_every_layer(exact):
    from tests.test_csa2_routing import tiny_config
    from distillkit.models import Qwen35WidenedForCausalLM

    config = tiny_config(mla_enabled=False, residual_stream_blend=1.0,
                         residual_stream_norm_mode="exact" if exact else "fast")
    model = Qwen35WidenedForCausalLM(config)
    routes = [module for module in model.modules()
              if isinstance(module, HyperConnection)]
    assert routes
    assert all((route.norm_mode == "exact") is exact for route in routes)


def test_a_from_scratch_run_still_trains_on_the_fast_path():
    """Cheap is only useful if the gradients still arrive."""
    from tests.test_csa2_routing import tiny_config
    from distillkit.models import Qwen35WidenedForCausalLM

    torch.manual_seed(0)
    config = tiny_config(mla_enabled=False, residual_stream_blend=1.0,
                         residual_stream_norm_mode="fast")
    config._attn_implementation = "eager"
    model = Qwen35WidenedForCausalLM(config).float()
    model.train()
    tokens = torch.randint(1, 64, (2, 16))
    model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                use_cache=False).last_hidden_state.pow(2).mean().backward()
    routing = [p for n, p in model.named_parameters()
               if any(k in n for k in ("W_down", "W_up", "W_write", "branch_gain_delta"))]
    assert routing
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in routing)
    assert any(p.grad.abs().sum() > 0 for p in routing)


def _modes_on(states, gain=0.1):
    """Each mode's read, with a *trained* branch gain.

    `branch_gain_delta` initialises to zero, so the gain is exactly 1.0 and multiplying
    by it rounds nothing -- every mode then agrees bitwise even in bf16, and a comparison
    at initialisation would measure nothing at all. The modes only separate once the gain
    has moved, which is the state every step after the first is in.
    """
    built = {}
    for mode in HyperConnection.NORM_MODES:
        torch.manual_seed(1)
        route = HyperConnection(hidden_size=HIDDEN, num_branches=BRANCHES, lowrank=8,
                                blend=1.0, norm_mode=mode).to(states.dtype)
        with torch.no_grad():
            generator = torch.Generator().manual_seed(7)
            route.branch_gain_delta.copy_(
                torch.randn(BRANCHES, HIDDEN, generator=generator) * gain)
            built[mode] = route._normalize(states)
    scale = built["exact"].float().abs().max()
    return {mode: ((built["exact"].float() - value.float()).abs().max() / scale).item()
            for mode, value in built.items()}


def test_the_cheap_modes_cost_nothing_in_float32():
    """`fused` loses nothing when the arithmetic is already fp32 -- the cost is a cast.

    `F.rms_norm` computes in the input dtype, so at fp32 it is the same arithmetic as the
    exact form and matches it outright. Every figure about `fused` losing precision is a
    statement about bf16, and only about bf16.
    """
    torch.manual_seed(0)
    relative = _modes_on(torch.randn(2, 16, BRANCHES, HIDDEN))
    assert relative["exact"] == 0.0
    assert relative["fast"] < 1e-5, relative
    assert relative["fused"] < 1e-5, relative


def test_fused_is_a_coarser_approximation_than_fast_in_bfloat16():
    """And bf16 is what training runs in, which is why there are three modes.

    The fused kernel normalises in the input dtype, so the normalised value is rounded to
    bf16 before the per-branch gain multiplies it -- one extra rounding that the exact
    form avoids by keeping the product in fp32 until the end.
    """
    torch.manual_seed(0)
    relative = _modes_on(torch.randn(2, 16, BRANCHES, HIDDEN).bfloat16())
    assert relative["exact"] == 0.0
    assert relative["fast"] < relative["fused"], relative
    assert relative["fused"] < 5e-2, relative


def test_every_mode_is_identical_while_the_gain_is_still_zero():
    """Which is why a smoke test at step 0 cannot tell them apart."""
    torch.manual_seed(0)
    relative = _modes_on(torch.randn(2, 16, BRANCHES, HIDDEN).bfloat16(), gain=0.0)
    assert set(relative.values()) == {0.0}, relative


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="unknown norm_mode"):
        HyperConnection(hidden_size=HIDDEN, num_branches=BRANCHES, lowrank=8,
                        norm_mode="quick")


@pytest.mark.parametrize("mode", ["fast", "fused"])
def test_a_from_scratch_run_trains_in_every_cheap_mode(mode):
    from tests.test_csa2_routing import tiny_config
    from distillkit.models import Qwen35WidenedForCausalLM

    torch.manual_seed(0)
    config = tiny_config(mla_enabled=False, residual_stream_blend=1.0,
                         residual_stream_norm_mode=mode)
    config._attn_implementation = "eager"
    model = Qwen35WidenedForCausalLM(config).float()
    model.train()
    tokens = torch.randint(1, 64, (2, 16))
    model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                use_cache=False).last_hidden_state.pow(2).mean().backward()
    routing = [p for n, p in model.named_parameters()
               if any(k in n for k in ("W_down", "W_up", "W_write", "branch_gain_delta"))]
    assert routing
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in routing)
    assert any(p.grad.abs().sum() > 0 for p in routing)


def test_recipient_conversion_refuses_an_approximate_norm():
    """The conversion's only claim is that the read *is* the recipient's sublayer."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    norm = Qwen3_5RMSNorm(HIDDEN, eps=1e-6)
    for mode in ("fast", "fused"):
        route = HyperConnection(hidden_size=HIDDEN, num_branches=BRANCHES, lowrank=8,
                                blend=1.0, norm_mode=mode)
        with pytest.raises(ValueError, match="norm_mode='exact'"):
            route.recipient_initialize(norm)
    exact = HyperConnection(hidden_size=HIDDEN, num_branches=BRANCHES, lowrank=8,
                            blend=1.0, norm_mode="exact")
    exact.recipient_initialize(norm)
    assert exact.recipient_initialized


def test_the_run_identity_separates_everything_that_changes_training():
    import sys
    sys.path.insert(0, "scratch/dense_gr")
    from benchmark import variant_tag

    seen = {variant_tag(*args) for args in (
        ("3:1", 0.0, "exact", 0), ("3:1", 1.0, "exact", 0), ("3:1", 0.5, "exact", 0),
        ("3:1", 1.0, "fused", 0), ("3:1", 1.0, "fast", 0), ("3:1", 1.0, "exact", 1),
        ("1:1", 1.0, "exact", 0))}
    # Seven configurations that differ in what trains must have seven names; the old tag
    # collapsed blend 1.0 with 0.5, and every norm mode with every other.
    assert len(seen) == 7, sorted(seen)
    assert variant_tag("3:1", 0.0) == "r3-1-nogr"
    assert variant_tag("3:1", 1.0) == "r3-1-gr"
