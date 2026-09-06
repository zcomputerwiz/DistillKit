"""Gates on the gated residual (spec §3b).

The two properties that matter are in tension, so both are pinned here:
identity must be *exact* (gate 3 compares logits bitwise), and the gates must
stay off the saturated tail where they would receive no gradient.
"""

import pytest
import torch

from distillkit.gated_residual import GatedResidual


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_branches", [1, 2, 4, 8])
def test_identity_is_bit_exact_at_init(dtype, num_branches):
    """Not 'close to' identity -- bit-identical, in every dtype the run might use."""
    torch.manual_seed(0)
    module = GatedResidual(64, num_branches=num_branches, dtype=dtype).eval()
    x = torch.randn(3, 17, 64, dtype=dtype)
    h = torch.randn(3, 17, 64, dtype=dtype)
    assert torch.equal(module(x, h), x)
    assert torch.equal(module(x, None), x)


def test_identity_holds_for_hostile_inputs():
    """Zero branches must survive values that would expose an approximate identity."""
    torch.manual_seed(0)
    module = GatedResidual(32, num_branches=4).eval()
    for x in [
        torch.zeros(2, 5, 32),
        torch.full((2, 5, 32), 1e4),
        torch.full((2, 5, 32), -1e4),
        torch.full((2, 5, 32), 1e-8),
        torch.randn(2, 5, 32) * 1e3,
    ]:
        assert torch.equal(module(x, x), x)


def test_gates_start_unsaturated():
    """sigmoid(0)=0.5 is the maximum-gradient point. Explicitly not the b=+6.0 tail."""
    torch.manual_seed(0)
    module = GatedResidual(128, num_branches=4)
    x = torch.randn(4, 32, 128)
    h = torch.randn(4, 32, 128)
    logits = module.W_x(x) + module.W_h(h)
    gates = torch.sigmoid(logits)

    assert 0.45 < gates.mean().item() < 0.55, "gates should sit near 0.5 at init"
    # sigmoid'(z) = g(1-g); at 0.5 this is 0.25, the maximum.
    slope = (gates * (1 - gates)).mean().item()
    assert slope > 0.24, f"gates are saturated at init (mean slope {slope:.4f})"

    # And prove the alternative really is bad, so the comment is not folklore:
    saturated = torch.sigmoid(torch.tensor(6.0))
    assert saturated.item() != 1.0, "sigmoid(6) is not identity"
    assert (saturated * (1 - saturated)).item() < 0.003, "sigmoid(6) is in the dead tail"


def test_branch_weights_receive_gradient_despite_zero_init():
    """The whole design rests on this: zero-init != zero-gradient.

    dL/dW = delta (x) input, and the input is nonzero, so a zero weight still moves.
    """
    torch.manual_seed(0)
    module = GatedResidual(64, num_branches=4)
    x = torch.randn(2, 8, 64)
    h = torch.randn(2, 8, 64)
    module(x, h).square().mean().backward()

    for index, branch in enumerate(module.branches):
        grad = branch.weight.grad
        assert grad is not None, f"branch {index} got no grad"
        assert grad.norm().item() > 0, f"branch {index} grad is identically zero"


def test_gates_are_dead_on_step_one_and_alive_on_step_two():
    """Documents the one real cost of exact-identity init, rather than hiding it."""
    torch.manual_seed(0)
    module = GatedResidual(64, num_branches=2)
    x = torch.randn(2, 8, 64)
    h = torch.randn(2, 8, 64)

    module(x, h).square().mean().backward()
    assert module.W_x.weight.grad.norm().item() == 0.0, (
        "expected zero gate gradient while branches are still exactly zero"
    )

    # Nudge a branch off zero, as the first optimizer step would.
    with torch.no_grad():
        module.branches[0].weight.normal_(std=0.01)
    module.zero_grad()
    module(x, h).square().mean().backward()
    assert module.W_x.weight.grad.norm().item() > 0, (
        "gates must start learning once branches leave zero"
    )


def test_single_branch_is_a_true_passthrough():
    """num_branches=1 is the ablation control arm; it must add no parameters."""
    module = GatedResidual(64, num_branches=1)
    assert sum(p.numel() for p in module.parameters()) == 0
    x = torch.randn(2, 4, 64)
    assert torch.equal(module(x, x), x)
    assert module.gate_report() == {}


def test_gate_report_shows_movement_off_identity():
    """§3b wants to know whether GR is dead weight by the end of the run."""
    torch.manual_seed(0)
    module = GatedResidual(64, num_branches=3).train()
    module(torch.randn(2, 8, 64), torch.randn(2, 8, 64))

    at_init = module.gate_report()
    assert at_init["gated_residual/branch_1_weight_norm"] == 0.0
    assert at_init["gated_residual/branch_2_weight_norm"] == 0.0
    assert at_init["gated_residual/W_x_bias_absmax"] == 0.0
    assert at_init["gated_residual/gate_saturation"] < 0.05, "gates saturated at init"

    with torch.no_grad():
        for branch in module.branches:
            branch.weight.normal_(std=0.1)
    module(torch.randn(2, 8, 64), torch.randn(2, 8, 64))
    trained = module.gate_report()
    assert trained["gated_residual/branch_1_weight_norm"] > 0.0


def test_scalar_gate_variant_also_exact():
    module = GatedResidual(64, num_branches=4, per_channel_gate=False).eval()
    x = torch.randn(2, 6, 64)
    assert torch.equal(module(x, x), x)
    report_params = sum(p.numel() for p in module.parameters())
    per_channel = sum(p.numel() for p in GatedResidual(64, num_branches=4).parameters())
    assert report_params < per_channel


def test_shapes_survive_odd_ranks():
    """Trainer hands [B, T, H]; keep 2D working so unit-testing a layer is easy."""
    module = GatedResidual(16, num_branches=3).eval()
    for shape in [(16,), (4, 16), (2, 3, 16), (2, 3, 4, 16)]:
        x = torch.randn(*shape)
        assert module(x, x).shape == x.shape
