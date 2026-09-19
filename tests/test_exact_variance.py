"""The cheap variance is available, and off wherever a donor has to be reproduced.

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
    assert route.exact_variance is True


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
                                blend=1.0, exact_variance=exact)
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
                         residual_stream_exact_variance=exact)
    model = Qwen35WidenedForCausalLM(config)
    routes = [module for module in model.modules()
              if isinstance(module, HyperConnection)]
    assert routes
    assert all(route.exact_variance is exact for route in routes)


def test_a_from_scratch_run_still_trains_on_the_fast_path():
    """Cheap is only useful if the gradients still arrive."""
    from tests.test_csa2_routing import tiny_config
    from distillkit.models import Qwen35WidenedForCausalLM

    torch.manual_seed(0)
    config = tiny_config(mla_enabled=False, residual_stream_blend=1.0,
                         residual_stream_exact_variance=False)
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
