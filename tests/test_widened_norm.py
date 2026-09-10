"""The widened read's norm is a hand-written autograd Function, so it needs pinning.

`Qwen3_5RMSNorm` materialises four full-width fp32 tensors for a bf16 input and
autograd holds two of them until backward -- about 670 MB per widened layer at batch
2 x 4096, the largest item in that layer's recompute working set. `branch_norm` keeps
only the bf16 input and a per-row reciprocal standard deviation and differentiates the
closed form instead.

Two things therefore have to hold. The forward must still be *bit-identical* to the
module, because the whole widening rests on being exactly the identity at
initialisation. The backward is allowed to differ in rounding, but not in value.

A previous substitution of a norm implementation in this project (torch's `nn.RMSNorm`
for the zero-centred one) was identical at initialisation and silently unable to learn
in bf16, so "it matches at init" is explicitly not what these check.
"""

import pytest
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

from distillkit.widened_residual import branch_norm

WIDTH = 128


def reference(x, norm, gain_delta):
    return norm(x) * (1 + gain_delta)


def _norm(width=WIDTH, trained=True, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    norm = Qwen3_5RMSNorm(width).to(dtype)
    if trained:  # a checkpoint's weight is a deviation from zero, not zero
        with torch.no_grad():
            norm.weight.normal_(0, .05)
    return norm


@pytest.mark.parametrize("scale", [1.0, 40.0, 0.01])
@pytest.mark.parametrize("trained", [False, True])
def test_forward_is_bit_identical_at_zero_gain(scale, trained):
    """Zero gain is the initialisation the identity proof depends on."""
    torch.manual_seed(1)
    x = (torch.randn(3, 17, WIDTH) * scale).to(torch.bfloat16)
    norm = _norm(trained=trained)
    gain = torch.zeros(WIDTH, dtype=torch.bfloat16)

    got = branch_norm(x, norm, gain)
    want = reference(x, norm, gain)
    assert torch.equal(got, want), (got.float() - want.float()).abs().max().item()


def test_forward_is_bit_identical_on_outlier_channels():
    """The channels LLM.int8() cares about are where a reordered cast shows up."""
    torch.manual_seed(2)
    x = torch.randn(2, 9, WIDTH)
    x[..., ::16] *= 60
    x = x.to(torch.bfloat16)
    norm = _norm()
    got = branch_norm(x, norm, torch.zeros(WIDTH, dtype=torch.bfloat16))
    want = reference(x, norm, torch.zeros(WIDTH, dtype=torch.bfloat16))
    assert torch.equal(got, want)


def test_forward_tracks_the_module_once_the_gain_has_trained():
    """Folding the gain into the weight removes one rounding, so this is close
    rather than exact -- but it must not drift."""
    torch.manual_seed(3)
    x = torch.randn(2, 11, WIDTH, dtype=torch.bfloat16)
    norm = _norm()
    gain = (torch.randn(WIDTH) * .1).to(torch.bfloat16)
    torch.testing.assert_close(branch_norm(x, norm, gain).float(),
                               reference(x, norm, gain).float(), rtol=8e-3, atol=8e-3)


@pytest.mark.parametrize("width", [4, 128])
def test_backward_matches_autograd_through_the_module(width):
    """The closed form against the module's own graph, in fp32 where both are exact
    enough to compare tightly."""
    torch.manual_seed(4)
    norm = _norm(width, dtype=torch.float32)
    gain_source = torch.randn(width) * .1

    results = []
    for fn in (branch_norm, reference):
        torch.manual_seed(4)  # both arms must see the same input, not just the same seed once
        x = torch.randn(2, 7, width, requires_grad=True)
        gain = gain_source.clone().requires_grad_(True)
        weight = norm.weight.detach().clone().requires_grad_(True)
        local = Qwen3_5RMSNorm(width)
        local.weight = torch.nn.Parameter(weight)
        (fn(x, local, gain) * torch.randn(2, 7, width, generator=
            torch.Generator().manual_seed(5))).sum().backward()
        results.append((x.grad, gain.grad, local.weight.grad))

    for got, want, name in zip(results[0], results[1], ("input", "gain", "weight")):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6, msg=lambda m: name + ": " + m)


def test_backward_is_correct_in_double_precision():
    """gradcheck on the Function itself, where an algebra slip cannot hide."""
    from distillkit.widened_residual import _BranchNorm

    torch.manual_seed(6)
    x = torch.randn(2, 3, 5, dtype=torch.double, requires_grad=True)
    gain = torch.randn(5, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, g: _BranchNorm.apply(a, g, 1e-6),
                                    (x, gain), eps=1e-6, atol=1e-9)


def test_non_contiguous_branch_views_are_accepted():
    """`read` passes `states.unbind(-2)`, which are strided views, not copies."""
    torch.manual_seed(7)
    states = torch.randn(2, 5, 3, WIDTH, dtype=torch.bfloat16, requires_grad=True)
    norm = _norm()
    gain = torch.zeros(WIDTH, dtype=torch.bfloat16)
    for branch in states.unbind(-2):
        assert not branch.is_contiguous()
        assert torch.equal(branch_norm(branch, norm, gain), reference(branch, norm, gain))
    torch.stack([branch_norm(b, norm, gain) for b in states.unbind(-2)], -2).sum().backward()
    assert torch.isfinite(states.grad).all() and states.grad.abs().sum() > 0


def test_an_unrecognised_norm_falls_back_to_the_module():
    """A different norm must degrade to the slow path, never to wrong arithmetic."""
    class _Other(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.full((WIDTH,), 3.0))

        def forward(self, x):
            return x * self.weight

    other = _Other()
    x = torch.randn(2, 4, WIDTH)
    gain = torch.zeros(WIDTH)
    assert torch.equal(branch_norm(x, other, gain), other(x) * (1 + gain))


def test_the_weight_can_still_move_at_a_realistic_step():
    """The norm weight is a trainable backbone parameter in stage 2; a Function that
    returned no gradient for it would freeze it silently, which is the failure this
    project has already had twice."""
    norm = _norm()
    x = torch.randn(2, 6, WIDTH, dtype=torch.bfloat16)
    gain = torch.zeros(WIDTH, dtype=torch.bfloat16, requires_grad=True)
    branch_norm(x, norm, gain).float().pow(2).sum().backward()
    assert norm.weight.grad is not None and norm.weight.grad.abs().sum() > 0
    assert gain.grad is not None and gain.grad.abs().sum() > 0


def _autograd_held_bytes(fn, x, norm, gain):
    """What autograd keeps alive until backward, counted rather than estimated."""
    seen = {}

    def pack(tensor):
        seen[id(tensor)] = tensor.numel() * tensor.element_size()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn(x, norm, gain).sum().backward()
    return sum(seen.values())


def test_it_holds_a_fifth_of_what_the_module_holds():
    """The reason this Function exists, measured rather than argued.

    The module keeps the fp32 cast of the input (for the square) and the normalised
    value (for the weight multiply): ten bytes per element of a two-byte input. This
    keeps the bf16 input the residual stream is holding anyway, plus one fp32 scalar
    per row, so the marginal cost is the rows alone."""
    torch.manual_seed(8)
    x = torch.randn(2, 64, WIDTH, dtype=torch.bfloat16, requires_grad=True)
    norm = _norm()
    gain = torch.zeros(WIDTH, dtype=torch.bfloat16, requires_grad=True)

    module = _autograd_held_bytes(reference, x, norm, gain) / x.numel()
    fused = _autograd_held_bytes(branch_norm, x, norm, gain) / x.numel()

    assert module > 9.9, module          # fp32 copy + fp32 normalised value + the bf16 input
    assert fused < 2.2, fused            # the bf16 input, and per-row reciprocals
    assert module / fused > 4, (module, fused)


def test_float64_falls_back_to_the_module():
    """`Qwen3_5RMSNorm` computes from `x.float()`, discarding half a double's mantissa.
    Matching it exactly matters more than the memory this saves, and nothing trains in
    float64 -- but gradcheck runs there, so the Function keeps its double support."""
    norm = _norm(dtype=torch.float64)
    x = torch.randn(2, 5, WIDTH, dtype=torch.float64)
    gain = torch.zeros(WIDTH, dtype=torch.float64)
    assert torch.equal(branch_norm(x, norm, gain), reference(x, norm, gain))


def test_a_bare_hidden_vector_reduces_correctly():
    """`sum(dim=())` reduces everything rather than nothing, so an input with no leading
    dimensions collapsed the gain gradient to a scalar and failed autograd's shape check."""
    from distillkit.widened_residual import _BranchNorm

    x = torch.randn(9, dtype=torch.double, requires_grad=True)
    gain = torch.randn(9, dtype=torch.double, requires_grad=True)
    _BranchNorm.apply(x, gain, 1e-6).sum().backward()
    assert gain.grad.shape == gain.shape and torch.isfinite(gain.grad).all()
    assert x.grad.shape == x.shape


def test_second_derivatives_are_refused_rather_than_wrong():
    """rstd is computed inside forward and carries no graph, so a second derivative
    through it would be silently wrong. Raising beats returning a plausible number."""
    from distillkit.widened_residual import _BranchNorm

    x = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
    gain = torch.randn(4, dtype=torch.double, requires_grad=True)
    out = _BranchNorm.apply(x, gain, 1e-6)
    (grad,) = torch.autograd.grad(out.sum(), x, create_graph=True)
    with pytest.raises(RuntimeError):
        torch.autograd.grad(grad.sum(), x)
