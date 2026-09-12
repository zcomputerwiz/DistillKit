"""A private residual lane for local/structural information, and learned reads from it.

The offload branch closed negative: removing whitespace-selection work from an
already-trained single residual stream does not improve content
(``scratch/ple_forensics/RESULTS.md``). What that rules out is *loss removal* as the
mechanism. What it leaves standing is representational separation -- PLE may only pay
when local information has its own state, so it never contaminates the residual the
backbone uses for content, and the network learns when to read it.

This is the smallest student-native version of that. The backbone keeps its ordinary
residual ``h``. A second lane ``m`` carries the sidecar's write and nothing else::

    m  =  PLE(h_l, features)              # written once, at the sidecar's layer
    h  <-  h + s_l * g_l(h) * R_l(m)      # read at each instrumented layer

``g_l`` is the sidecar's own admission criterion, a dot product against a learned
direction, compressed by the signed square root and squashed. It reads the current stream
and nothing else: no token class, no target, no oracle.

**Only one thing is zero-initialised, and it is the write, not the read.** The obvious
design puts ``s_l = 0`` as well, so the read is manifestly inert at step 0. That
deadlocks. The contribution is ``s_l * g_l(h) * R_l(m)``: with the sidecar's
``value_proj`` at zero the lane carries ``m = 0`` and ``s_l`` receives no gradient, and
with ``s_l = 0`` the lane is the value's only path to the loss so ``value_proj`` receives
none either. Both sit at exactly zero for the whole run -- measured four steps into the
first version of this module, every read scale still 0.0.

``s_l`` therefore starts at 1, and the model is *still* bitwise stock at initialisation,
because ``m`` is exactly zero until ``value_proj`` moves. The read then grows smoothly
with the lane rather than switching on: nothing renormalises the lane on the way in, so a
small write stays a small read.

**The gate direction is not zero-initialised, and that is deliberate.** ``signed_sqrt``
has ``sign(0) == 0`` and a clamp that flattens ``abs()`` near the origin, so a direction
at exactly zero receives exactly zero gradient and the gate would sit frozen at 0.5 for
the whole run -- measured in ``scratch/table_capacity_probe.py``, whose first gate
ablation compared three constant rescalings for that reason. A small random direction
costs nothing at initialisation because ``s_l`` is zero anyway.

``R_l`` defaults to the identity: the lane already lives in hidden space, because the
sidecar's ``value_proj`` writes there, and an identity read is the minimum mechanism that
still asks the question. ``project=True`` gives each read its own identity-initialised
matrix, which is the first thing to vary if the identity read is what fails.
"""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = ["MemoryRead"]


class MemoryRead(nn.Module):
    """``h -> h + s * sigmoid(signed_sqrt(<norm(h), w>)) * R(m)``, zero at initialisation."""

    def __init__(
        self,
        hidden_size: int,
        *,
        project: bool = False,
        scale_init: float = 1.0,
        gate_init_std: float = 0.02,
        eps: float = 1e-6,
    ):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if gate_init_std <= 0:
            raise ValueError("gate_init_std must be positive; a zero direction gets no gradient")
        self.hidden_size = hidden_size
        self.eps = eps
        # Not zero -- see the module docstring. Zero here *and* zero in the sidecar's
        # value projection is a product of two zeros, and neither factor can leave.
        self.scale = nn.Parameter(torch.full((1,), float(scale_init)))
        self.direction = nn.Parameter(torch.randn(hidden_size) * gate_init_std)
        self.proj = None
        if project:
            self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
            with torch.no_grad():
                self.proj.weight.copy_(torch.eye(hidden_size))
        #: Detached per-token diagnostics from the last forward, for the question the
        #: pilot exists to answer: does joint training learn a *token-dependent* read, or
        #: one constant admission for everything? Kept as tensors so nothing synchronises.
        self.last_alpha: torch.Tensor | None = None
        self.last_ratio: torch.Tensor | None = None

    def alpha(self, stream: torch.Tensor) -> torch.Tensor:
        """Per-token read strength ``s * g(h)``, shape ``[batch, seq, 1]``."""
        normed = stream.float()
        normed = normed * torch.rsqrt(normed.pow(2).mean(-1, keepdim=True) + self.eps)
        raw = (normed @ self.direction.float()).unsqueeze(-1) / math.sqrt(self.hidden_size)
        gate = torch.sigmoid(raw.abs().clamp_min(1e-6).sqrt() * raw.sign())
        return self.scale.float() * gate

    def forward(self, stream: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """The contribution to add to ``stream``; the caller adds it.

        Computed in this module's own dtype rather than the stream's. These parameters
        are held in fp32 while the backbone is bf16, because ``direction`` is initialised
        at std 0.02, where bf16 spacing is 1.6e-4 -- every update at this project's rates
        would round away and the gate would be frozen at its initialisation. Same failure
        mode ``_PLERMSNorm`` exists to avoid.
        """
        memory = memory.to(device=stream.device, dtype=self.scale.dtype)
        read = self.proj(memory) if self.proj is not None else memory
        alpha = self.alpha(stream)
        contribution = alpha.to(read.dtype) * read
        with torch.no_grad():
            self.last_alpha = alpha.detach().squeeze(-1).float()
            self.last_ratio = (contribution.detach().float().norm(dim=-1)
                               / stream.detach().float().norm(dim=-1).clamp_min(1e-9))
        return contribution.to(stream.dtype)


def _demo() -> None:
    torch.manual_seed(0)
    read = MemoryRead(16)
    stream = torch.randn(2, 5, 16)
    memory = torch.randn(2, 5, 16)
    empty = torch.zeros_like(memory)

    # An empty lane reads as exactly nothing, which is what keeps the model stock while
    # the sidecar's value projection is still at zero.
    assert torch.equal(read(stream, empty), torch.zeros_like(stream))

    # The deadlock this initialisation exists to avoid: with an empty lane the read's own
    # scale has no gradient either, so a zero-initialised scale would never leave zero.
    read(stream, empty).sum().backward()
    assert read.scale.grad.abs().item() == 0.0

    # Once the lane carries anything, both the strength and the direction train.
    read.zero_grad(set_to_none=True)
    read(stream, memory).sum().backward()
    assert read.scale.grad.abs().item() > 0
    assert read.direction.grad.abs().max().item() > 0

    # A projecting read starts as the identity, so `project` changes capacity, not the
    # initial function.
    projecting = MemoryRead(16, project=True)
    with torch.no_grad():
        plain = MemoryRead(16)
        plain.direction.copy_(projecting.direction)
        assert torch.allclose(projecting(stream, memory), plain(stream, memory), atol=1e-6)
    print("memory_lane demo ok")


if __name__ == "__main__":
    _demo()
