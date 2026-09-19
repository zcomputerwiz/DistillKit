"""Normalizing every branch at once must equal normalizing them one at a time.

The per-branch loop was not load-bearing -- RMSNorm reduces over the last dimension, so
the branch axis is just more leading shape. What made the batched form wrong before was
`_BranchNorm.backward` reducing the gain gradient over every dimension but the last,
which sums the branch axis away and hands all branches the same gradient. That is silent:
the forward is identical, the shapes still check, and only the per-branch gains stop
being per-branch.
"""

import pytest
import torch

from distillkit.experimental.widened_residual import _BranchNorm

BRANCHES, HIDDEN, EPS = 4, 8, 1e-6


def loop(states, gain):
    """What the code did before: one call per branch, then a stack."""
    return torch.stack([
        _BranchNorm.apply(branch, 1.0 + row, EPS)
        for branch, row in zip(states.unbind(-2), gain)
    ], dim=-2)


@pytest.fixture
def inputs():
    torch.manual_seed(0)
    states = torch.randn(3, 5, BRANCHES, HIDDEN, dtype=torch.float64,
                         requires_grad=True)
    gain = torch.randn(BRANCHES, HIDDEN, dtype=torch.float64) * 0.1
    return states, gain.requires_grad_(True)


def test_batched_forward_matches_the_loop(inputs):
    states, gain = inputs
    batched = _BranchNorm.apply(states, 1.0 + gain, EPS)
    assert torch.allclose(batched, loop(states, gain), atol=1e-12, rtol=1e-12)


def test_each_branch_keeps_its_own_gain_gradient(inputs):
    """The failure the old reduction caused: identical gradients on every branch."""
    states, gain = inputs
    _BranchNorm.apply(states, 1.0 + gain, EPS).pow(2).mul(
        torch.arange(1.0, BRANCHES + 1, dtype=torch.float64).view(-1, 1)).sum().backward()
    batched_states, batched_gain = states.grad.clone(), gain.grad.clone()
    states.grad = gain.grad = None

    loop(states, gain).pow(2).mul(
        torch.arange(1.0, BRANCHES + 1, dtype=torch.float64).view(-1, 1)).sum().backward()
    assert torch.allclose(batched_states, states.grad, atol=1e-12, rtol=1e-12)
    assert torch.allclose(batched_gain, gain.grad, atol=1e-12, rtol=1e-12)
    # The weighting above makes every branch's gradient distinct, so a reduction that
    # collapsed the branch axis would show up as equal rows rather than merely wrong ones.
    assert not torch.allclose(batched_gain[0], batched_gain[1])


def test_a_one_dimensional_gain_still_reduces_the_same_way():
    """The `[hidden]` case has to be untouched; it is what every other caller passes."""
    torch.manual_seed(0)
    x = torch.randn(3, 5, HIDDEN, dtype=torch.float64, requires_grad=True)
    gain = torch.randn(HIDDEN, dtype=torch.float64).requires_grad_(True)
    _BranchNorm.apply(x, gain, EPS).pow(2).sum().backward()
    assert gain.grad.shape == (HIDDEN,)
    assert torch.isfinite(gain.grad).all()
