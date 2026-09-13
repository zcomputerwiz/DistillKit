"""Replicated parameter gradient synchronization and gradient norm clipping.

For tensor-parallel models where certain parameters (e.g. norms) are replicated
across ranks rather than sharded, each copy accumulates a partial gradient during
backward. These helpers all-reduce the partial gradients and ensure that gradient
norm clipping counts each replicated parameter exactly once.
"""

from __future__ import annotations

import torch
from torch import nn

from distillkit.parallel.collectives import all_reduce

__all__ = [
    "replicated_parameter_groups",
    "sync_replicated_gradients",
    "clip_grad_norm",
]


def replicated_parameter_groups(module: nn.Module):
    """Yield tuples of replicated parameter copies across devices."""
    for child in module.modules():
        groups = getattr(child, "replicated_parameters", None)
        if groups is not None:
            yield from zip(*groups())


@torch.no_grad()
def sync_replicated_gradients(module: nn.Module) -> int:
    """All-reduce gradients of parameters replicated across ranks.

    A replicated parameter sees only its rank's share of the loss, so each holds a
    partial gradient. Summing them is what makes the replica equivalent to the
    unsharded parameter. Returns how many parameter groups were reduced.

    Call this after backward and before the optimizer step. Omitting it does not
    raise; the norm simply trains against a fraction of its gradient.
    """
    reduced = 0
    for group in replicated_parameter_groups(module):
        grads = [p.grad for p in group]
        if all(g is None for g in grads):
            continue
        if any(g is None for g in grads):
            raise RuntimeError("A tensor-parallel replica is missing its gradient")
        totals = all_reduce(grads)
        for parameter, total in zip(group, totals):
            parameter.grad = total.detach()
        reduced += 1
    return reduced


@torch.no_grad()
def clip_grad_norm(module: nn.Module, max_norm: float):
    """Count each replicated parameter once, then apply clipping to every copy."""
    groups = list(replicated_parameter_groups(module))
    duplicates = {id(p) for group in groups for p in group[1:]}
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in module.parameters() if id(p) not in duplicates], max_norm,
    )
    for group in groups:
        if group[0].grad is not None:
            for replica in group[1:]:
                replica.grad.copy_(group[0].grad.to(replica.device))
    return norm
