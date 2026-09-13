"""Multi-branch gated residual, initialized to an *exact* identity.

Spec §3b asks for a retrofitted gated residual whose init is exactly the identity
while leaving the gates unsaturated. Those two requirements pull against each other
if you try to satisfy both with the gate, which is why the naive ``bias = +6.0``
trick is wrong on both counts:

    sigmoid(6)  ~= 0.9975   -- close to identity, but not identity
    sigmoid'(6) ~= 0.0025   -- 100x less gradient than at the origin

You end up with an architecture that is neither exactly the model you started from
nor able to learn its way off that starting point. Runs like that look "stable" and
are actually inert.

The resolution is to put the zero in the *branches* rather than the gate:

    out = x + sum_k  g_k(x, h) * B_k(x, h)      with every B_k zero-initialized

Branch 0 is the residual stream itself, carried ungated, so identity is exact by
construction rather than approximate. The extra branches contribute exactly zero at
init because ``B_k`` is zero, not because a gate is pinned shut -- so the gates are
free to sit at the origin where their gradient is maximal.

This costs one thing, and it is worth naming: ``dL/dW_gate`` is proportional to
``B_k(x, h)``, so the gates receive no gradient on the very first step. They start
learning as soon as the branches move off zero. Branch weights themselves *do* get
gradient immediately (``dL/dB = delta (x) input``, and the input is nonzero) -- this
is measured, not assumed; see ``tests/test_gated_residual.py``.

``gate_report()`` exists because §3b asks for gate activations and ``W_x``/``W_h``
norms to be logged: if they have not moved off identity by the end of a run, the
gated residual is dead weight and should be dropped rather than carried into the
long run.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["GatedResidual", "GateReport"]


class GateReport(dict):
    """Plain dict of scalars, shaped for ``Trainer.log``."""


class GatedResidual(nn.Module):
    """``x -> x + sum_k g_k * B_k(x, h)``, exactly the identity at initialization.

    Args:
        hidden_size: width of the residual stream.
        num_branches: total branch count *including* the identity branch. ``1`` makes
            this module a no-op passthrough (useful as an ablation control).
        gate_init_std: std of the gate projections. Small but nonzero, so gates start
            near ``sigmoid(0) = 0.5`` -- the maximum-gradient point -- with enough
            asymmetry that different branches do not stay tied to each other.
        per_channel_gate: if True each branch gets a ``hidden_size``-wide gate; if
            False, one scalar gate per branch. Per-channel is strictly more expressive
            and costs ``2 * hidden_size`` params per branch.

    The ``h`` input is the auxiliary signal the gate should condition on -- in this
    fork, the n-gram sidecar output. Passing ``h=None`` gates on ``x`` alone.
    """

    def __init__(
        self,
        hidden_size: int,
        num_branches: int = 4,
        gate_init_std: float = 0.02,
        per_channel_gate: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_branches < 1:
            raise ValueError(f"num_branches must be >= 1, got {num_branches}")

        self.hidden_size = hidden_size
        self.num_branches = num_branches
        self.per_channel_gate = per_channel_gate
        self.num_gated_branches = num_branches - 1

        gate_width = hidden_size if per_channel_gate else 1
        factory = {"dtype": dtype} if dtype is not None else {}

        if self.num_gated_branches:
            # Branch transforms: zero-init. This is what makes identity exact.
            self.branches = nn.ModuleList(
                [
                    nn.Linear(hidden_size, hidden_size, bias=False, **factory)
                    for _ in range(self.num_gated_branches)
                ]
            )
            for branch in self.branches:
                nn.init.zeros_(branch.weight)

            # Gate projections: W_x reads the residual stream, W_h the aux signal.
            # Deliberately NOT zero -- a zero gate weight plus a zero branch would
            # leave the gate with no way to differentiate itself from its neighbours.
            self.W_x = nn.Linear(hidden_size, gate_width * self.num_gated_branches,
                                 bias=True, **factory)
            self.W_h = nn.Linear(hidden_size, gate_width * self.num_gated_branches,
                                 bias=False, **factory)
            nn.init.normal_(self.W_x.weight, std=gate_init_std)
            nn.init.normal_(self.W_h.weight, std=gate_init_std)
            # bias 0 -> sigmoid(0) = 0.5, maximum gradient. Explicitly not +6.0.
            nn.init.zeros_(self.W_x.bias)
        else:
            self.branches = nn.ModuleList()
            self.W_x = None
            self.W_h = None

        self._last_gate_mean: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        if not self.num_gated_branches:
            return x

        gate_source = x if h is None else h
        logits = self.W_x(x) + self.W_h(gate_source)
        gates = torch.sigmoid(logits)

        gate_width = self.hidden_size if self.per_channel_gate else 1
        gates = gates.unflatten(-1, (self.num_gated_branches, gate_width))

        # Stash for gate_report() without holding the graph.
        if self.training:
            self._last_gate_mean = gates.detach().mean(dim=tuple(range(gates.dim() - 2)))

        branch_input = x if h is None else h
        out = x
        for index, branch in enumerate(self.branches):
            out = out + gates[..., index, :] * branch(branch_input)
        return out

    @torch.no_grad()
    def gate_report(self, prefix: str = "gated_residual") -> GateReport:
        """Scalars for §3b's "has this moved off identity yet?" check.

        ``branch_weight_norm`` is the number to actually watch. It starts at exactly
        zero; if it is still ~zero when the run ends, the gated residual learned
        nothing and should be dropped rather than carried into the long run.
        """
        report = GateReport()
        if not self.num_gated_branches:
            return report

        report[f"{prefix}/W_x_norm"] = self.W_x.weight.float().norm().item()
        report[f"{prefix}/W_h_norm"] = self.W_h.weight.float().norm().item()
        report[f"{prefix}/W_x_bias_absmax"] = self.W_x.bias.float().abs().max().item()

        for index, branch in enumerate(self.branches):
            report[f"{prefix}/branch_{index + 1}_weight_norm"] = (
                branch.weight.float().norm().item()
            )

        if self._last_gate_mean is not None:
            means = self._last_gate_mean.float()
            for index in range(means.shape[0]):
                report[f"{prefix}/gate_{index + 1}_mean"] = means[index].mean().item()
            # Distance from 0.5 in either direction; ->0.5 means fully saturated.
            report[f"{prefix}/gate_saturation"] = (means - 0.5).abs().max().item()
        return report

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_branches={self.num_branches}, "
            f"per_channel_gate={self.per_channel_gate}"
        )
