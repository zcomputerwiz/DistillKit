"""The routing arithmetic, named and pinned separately from the module that calls it.

`_combine` is the whole elementwise tail of the widened read -- both gates, the
corrected value and the write weights. It is worth testing on its own because the
property everything else rests on lives here: at initialisation the correction term is
exactly zero and the write weights are exactly one, so the widened model reproduces the
unwidened one bit for bit. That has to hold structurally rather than by rounding.
"""

import torch

from distillkit.widened_residual import WidenedResidual, _combine, collapse_residual


def test_combine_is_the_identity_at_initialisation():
    torch.manual_seed(0)
    branches, width, index = 3, 8, 1
    route = WidenedResidual(width, num_branches=branches, lowrank=4, layer_idx=index)
    normalized = torch.randn(2, 5, branches, width)
    gate_logits = torch.randn(2, 5, branches * width)
    write_logits = torch.randn(2, 5, branches)

    value, weights = _combine(normalized, gate_logits, write_logits, route.read_offset,
                              route.lambda_read, route.write_offset, route.lambda_write,
                              route.read_index)

    assert torch.equal(value, normalized[..., index, :])
    assert torch.equal(weights, torch.ones_like(weights))


def test_combine_matches_the_arithmetic_written_longhand():
    """Once the routes have trained, against the expression spelled out."""
    torch.manual_seed(1)
    branches, width = 2, 6
    normalized = torch.randn(3, 4, branches, width)
    gate_logits = torch.randn(3, 4, branches * width)
    write_logits = torch.randn(3, 4, branches)
    read_offset = torch.randn(branches)
    write_offset = torch.randn(branches)
    lambda_read, lambda_write = torch.tensor(0.7), torch.tensor(-0.3)

    value, weights = _combine(normalized, gate_logits, write_logits, read_offset,
                              lambda_read, write_offset, lambda_write, 1)

    read_gate = torch.sigmoid(gate_logits).unflatten(-1, (branches, width))
    correction = read_offset.unsqueeze(-1) + lambda_read * read_gate
    torch.testing.assert_close(
        value, normalized[..., 1, :] + (correction * normalized).sum(-2))
    torch.testing.assert_close(
        weights, (1 + write_offset) + lambda_write * torch.sigmoid(write_logits))


def test_collapse_is_exact_when_the_branches_agree():
    """Expressed around branch zero so equal branches collapse to themselves exactly,
    which is what makes the widened logits bit-identical at initialisation."""
    branch = torch.randn(2, 3, 7, dtype=torch.bfloat16)
    states = branch.unsqueeze(-2).expand(2, 3, 4, 7)
    assert torch.equal(collapse_residual(states), branch)


def test_collapse_is_the_branch_mean():
    states = torch.randn(2, 3, 4, 5, dtype=torch.float64)
    torch.testing.assert_close(collapse_residual(states), states.mean(-2))
