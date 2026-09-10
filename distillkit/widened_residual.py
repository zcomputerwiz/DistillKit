"""Identity-initialized HC routing over a persistent widened residual stream.

Unlike the sidecar's additive GatedResidual, these branches carry state through
every attention and MLP block. There is no learned residual mixing matrix.
"""
from __future__ import annotations

import contextlib

import torch
from torch import nn
from torch.nn import functional as F


# Below this a saved tensor is bookkeeping (RNG state, scalars), not a stream.
_OFFLOAD_MIN_BYTES = 16 * 1024**2


@contextlib.contextmanager
def offload_stream_boundaries(peer, every: int = 2):
    """Park every ``every``-th checkpoint boundary tensor on the peer card.

    Widening multiplies the per-layer stream by the branch count, and the stream
    lives entirely on the tensor-parallel home card, so the home card pays the whole
    cost of the retrofit while the peer sits several gigabytes below it. Under
    non-reentrant checkpointing the only large tensors these hooks see in the layer
    loop are those per-layer boundaries -- recompute temporaries never leave the
    checkpoint frame -- so alternating them splits the stream evenly between cards.
    """
    if peer is None:
        yield
        return
    peer = torch.device(peer)
    seen = [0]

    def pack(tensor):
        if (tensor.device.type != "cuda" or tensor.device == peer
                or tensor.numel() * tensor.element_size() < _OFFLOAD_MIN_BYTES):
            return tensor
        seen[0] += 1
        if seen[0] % every:
            return tensor
        return tensor.device, tensor.to(peer)

    def unpack(payload):
        if isinstance(payload, tuple):
            home, tensor = payload
            return tensor.to(home)
        return payload

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield


def collapse_residual(states: torch.Tensor) -> torch.Tensor:
    """Mean over branches, expressed around branch zero for exact BF16 identity."""
    reference = states[..., 0, :]
    return reference + (states - reference.unsqueeze(-2)).mean(dim=-2)


class WidenedResidual(nn.Module):
    """HC static read/write plus zero-scaled, sigmoid data-dependent routing.

Read gates are per branch/channel, write gates per branch. All branches condition
both routes through a low-rank read bottleneck. Original pretrained norms belong
to the decoder and normalize each branch exactly once; no new norm is introduced.
Static read and write are stored as offsets from the identity route, which keeps
them representable in BF16; residual mixing is I.
Random dynamic projections allow the zero-initialized lambdas to learn immediately.
"""

    def __init__(self, hidden_size: int, num_branches: int = 2,
                 lowrank: int = 64, layer_idx: int = 0):
        super().__init__()
        if hidden_size < 1 or num_branches < 1 or lowrank < 1:
            raise ValueError("hidden_size, num_branches and lowrank must be positive")
        self.hidden_size, self.num_branches = hidden_size, num_branches
        self.read_index = layer_idx % num_branches
        # Offsets from the identity route, not the route itself: a parameter stored
        # at 1.0 in BF16 has a spacing of 0.0078 and cannot represent the ~1e-5 steps
        # this project trains with, so an all-ones static_write would sit frozen.
        self.read_offset = nn.Parameter(torch.zeros(num_branches))
        self.write_offset = nn.Parameter(torch.zeros(num_branches))
        self.lambda_read = nn.Parameter(torch.zeros(()))
        self.lambda_write = nn.Parameter(torch.zeros(()))
        # Branch-specific gains relative to the pretrained zero-centered RMSNorm.
        # Effective gain is (1 + original_weight) * (1 + branch_gain_delta).
        self.branch_gain_delta = nn.Parameter(torch.zeros(num_branches, hidden_size))
        self.W_down = nn.Linear(num_branches * hidden_size, lowrank, bias=False)
        self.W_up = nn.Linear(lowrank, num_branches * hidden_size, bias=False)
        self.W_write = nn.Linear(num_branches * hidden_size, num_branches, bias=False)
        self.reset_routing_parameters()

    @torch.no_grad()
    def reset_routing_parameters(self):
        self.read_offset.zero_()
        self.write_offset.zero_()
        self.lambda_read.zero_()
        self.lambda_write.zero_()
        self.branch_gain_delta.zero_()

    def read(self, states, norm):
        # One branch at a time keeps the norm's fp32 temporaries [B,T,d]-sized rather
        # than [B,T,n,d]; Windows has no expandable_segments, so a backward that
        # churns big odd-sized blocks strands gigabytes of reserved-but-unallocated
        # pool. The branch views are already contiguous in the normalized dimension.
        normalized = torch.stack([norm(branch) for branch in states.unbind(-2)], dim=-2)
        normalized = normalized * (1 + self.branch_gain_delta)
        flattened = normalized.flatten(-2)
        # Linear is homogeneous, so 1/n applies to the narrow output instead of a
        # second [B,T,n,d] copy of the input. n is a power of two in practice and the
        # scaling is exact; at initialisation both gates are multiplied by zero anyway.
        scale = 1 / self.num_branches
        read_gate = self.W_up(F.silu(self.W_down(flattened) * scale)).sigmoid_()
        read_gate = read_gate.unflatten(-1, (self.num_branches, self.hidden_size))
        write_gate = (self.W_write(flattened) * scale).sigmoid_()
        # The one-hot read is taken directly rather than as a weighted sum, so at
        # initialisation the correction term is exactly zero and the arithmetic is
        # bit-identical to the unwidened path.
        correction = self.read_offset.unsqueeze(-1) + self.lambda_read * read_gate
        value = normalized[..., self.read_index, :] + (correction * normalized).sum(-2)
        weights = (1 + self.write_offset) + self.lambda_write * write_gate
        return value.contiguous(), weights

    def write(self, states, output, weights):
        # addcmul fuses the broadcast product into the add, saving a [B,T,n,d] block.
        return torch.addcmul(states, weights.unsqueeze(-1), output.unsqueeze(-2))

    @torch.no_grad()
    def gate_report(self, prefix="residual_stream"):
        return {
            f"{prefix}/lambda_read": self.lambda_read.float().item(),
            f"{prefix}/lambda_write": self.lambda_write.float().item(),
            f"{prefix}/read_deviation": self.read_offset.float().norm().item(),
            f"{prefix}/write_deviation": self.write_offset.float().norm().item(),
        }
