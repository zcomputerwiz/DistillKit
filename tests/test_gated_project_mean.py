"""Folding the up-projection into the gated mean must not move the arithmetic.

The gate logits are one `[batch, tokens, branches * hidden]` tensor produced only to be
split per branch and then held until backward -- 268 MB at batch 64 by 1024, twenty times
over in a ten-layer stack. `_GatedProjectMean` projects one branch at a time so it is
never formed, and recomputes the slice it needs in backward from the `lowrank` code.

That moves a backward out of autograd and into hand-written code, which is the part worth
pinning: the composed version is the definition, and these check against it rather than
against a rederivation of the same algebra.
"""

import pytest
import torch

from distillkit.experimental.hyper_connection import _GatedMean, _GatedProjectMean

BRANCHES, HIDDEN, LOWRANK = 4, 8, 6
BATCH, TOKENS = 3, 5


def composed(normalized, code, weight):
    """What the route did before: project everything, then gate and mean."""
    logits = torch.nn.functional.linear(code, weight)
    return _GatedMean.apply(normalized, logits.unflatten(-1, (BRANCHES, HIDDEN)))


@pytest.fixture
def inputs():
    def make():
        # Seeded per call: the comparison is between two runs over the *same* inputs, and
        # seeding once outside would hand the second call different tensors and compare
        # gradients of different problems.
        generator = torch.Generator().manual_seed(0)
        normalized = torch.randn(BATCH, TOKENS, BRANCHES, HIDDEN, dtype=torch.float64,
                                 generator=generator)
        code = torch.randn(BATCH, TOKENS, LOWRANK, dtype=torch.float64,
                           generator=generator)
        weight = torch.randn(BRANCHES * HIDDEN, LOWRANK, dtype=torch.float64,
                             generator=generator) * 0.3
        return [t.requires_grad_(True) for t in (normalized, code, weight)]
    return make


def test_forward_matches_the_composed_version(inputs):
    normalized, code, weight = inputs()
    folded = _GatedProjectMean.apply(normalized, code, weight)
    assert torch.allclose(folded, composed(normalized, code, weight),
                          atol=1e-12, rtol=1e-12)


def test_backward_matches_the_composed_version(inputs):
    """Every gradient, including the one that now goes through a hand-written matmul."""
    a = inputs()
    _GatedProjectMean.apply(*a).pow(2).mul(
        torch.arange(1.0, HIDDEN + 1, dtype=torch.float64)).sum().backward()
    b = inputs()
    composed(*b).pow(2).mul(
        torch.arange(1.0, HIDDEN + 1, dtype=torch.float64)).sum().backward()
    for folded, reference, name in zip(a, b, ("normalized", "code", "weight")):
        assert torch.allclose(folded.grad, reference.grad, atol=1e-10, rtol=1e-10), name
        assert reference.grad.abs().sum() > 0, name + " reference gradient is zero"


def test_gradcheck(inputs):
    normalized, code, weight = inputs()
    assert torch.autograd.gradcheck(_GatedProjectMean.apply,
                                    (normalized, code, weight), eps=1e-6, atol=1e-8)


def test_the_gate_logits_are_never_materialised(inputs):
    """The point of the exercise: nothing `branches * hidden` wide is kept for backward."""
    normalized, code, weight = inputs()
    output = _GatedProjectMean.apply(normalized, code, weight)
    saved = {tuple(t.shape) for t in output.grad_fn.saved_tensors}
    assert tuple(code.shape) in saved, "the code is what backward recomputes from"
    assert not any(shape[-1] == BRANCHES * HIDDEN and len(shape) == 3 for shape in saved), (
        "a [batch, tokens, branches * hidden] tensor is being held: %s" % saved)


@pytest.mark.parametrize("branches", [1, 2, 4])
def test_branch_counts(branches):
    torch.manual_seed(0)
    normalized = torch.randn(2, 3, branches, HIDDEN, dtype=torch.float64,
                             requires_grad=True)
    code = torch.randn(2, 3, LOWRANK, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(branches * HIDDEN, LOWRANK, dtype=torch.float64,
                         requires_grad=True)
    logits = torch.nn.functional.linear(code, weight)
    reference = _GatedMean.apply(normalized, logits.unflatten(-1, (branches, HIDDEN)))
    assert torch.allclose(_GatedProjectMean.apply(normalized, code, weight), reference,
                          atol=1e-12, rtol=1e-12)
