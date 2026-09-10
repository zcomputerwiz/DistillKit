"""Identity-initialized HC routing over a persistent widened residual stream.

Unlike the sidecar's additive GatedResidual, these branches carry state through
every attention and MLP block. There is no learned residual mixing matrix.
"""
from __future__ import annotations

import contextlib

import torch
from torch import nn
from torch.nn import functional as F

from distillkit.fused import fused


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


class _BranchNorm(torch.autograd.Function):
    """RMSNorm and the per-branch gain, holding only what backward actually needs.

    `Qwen3_5RMSNorm` materialises four full-width fp32 tensors for a bf16 input, and
    autograd keeps two of them alive until backward: the fp32 copy of the input, for
    the square, and the normalised value, for the weight multiply. In the widened read
    that happens once per branch, twice per layer, and again on every checkpoint
    recompute. At batch 2 x 4096 with two branches it is about 670 MB held per widened
    layer while that layer's backward runs, which is the largest single item in the
    recompute working set and where the batch-4 run ran out of memory.

    This saves the bf16 input -- which the residual stream is holding anyway -- plus a
    `[..., 1]` reciprocal standard deviation, and differentiates the closed form:

        y = g * x * r,   r = rsqrt(mean(x^2) + eps)
        dL/dx = r*g*dy - (r^3 / n) * x * sum(dy * g * x)
        dL/dg = sum over leading dims of dy * x * r

    Forward is bit-identical to the module's own, because `x * rstd` promotes bf16 to
    fp32 inside the multiply kernel rather than materialising a cast copy first. The
    backward is the same quantity to fp32 rounding rather than the same sequence of
    operations. `tests/test_widened_norm.py` pins both against the module.
    """

    @staticmethod
    def forward(ctx, x, gain, eps):
        # fp32 is a floor, not the compute type: .float() on a float64 input would
        # silently discard half its mantissa, which gradcheck is entitled to notice.
        compute = torch.promote_types(x.dtype, torch.float32)
        variance = x.to(compute).pow(2).mean(-1, keepdim=True)
        rstd = torch.rsqrt(variance + eps)
        ctx.save_for_backward(x, gain, rstd)
        # bf16 * fp32 promotes inside the multiply kernel, so the cast copy of x that
        # the module holds until backward is never materialised at all.
        return ((x * rstd) * gain).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, gain, rstd = ctx.saved_tensors
        width = x.shape[-1]
        grad = grad_output.to(rstd.dtype)
        scaled = grad * gain
        inner = (scaled * x).sum(-1, keepdim=True)
        grad_x = rstd * scaled - (rstd.pow(3) / width) * inner * x
        grad_gain = None
        if ctx.needs_input_grad[1]:
            grad_gain = (grad * x * rstd).sum(dim=tuple(range(grad.ndim - 1)))
        return grad_x.to(x.dtype), grad_gain, None


def branch_norm(x, norm, gain_delta):
    """`norm(x) * (1 + gain_delta)`, computed without the module's fp32 copies.

    Falls back to the module whenever it is not the zero-centred RMSNorm this assumes,
    so a change upstream degrades to the slow path rather than to wrong arithmetic.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    if not isinstance(norm, Qwen3_5RMSNorm):
        return norm(x) * (1 + gain_delta)
    # Folding the gain into the weight drops one full-width multiply and one rounding.
    # At initialisation the gain is exactly zero, so this is exactly `1 + weight`.
    gain = (1.0 + norm.weight.float()) * (1.0 + gain_delta.float())
    return _BranchNorm.apply(x, gain, norm.eps)


@fused
def collapse_residual(states: torch.Tensor) -> torch.Tensor:
    """Mean over branches, expressed around branch zero for exact BF16 identity."""
    reference = states[..., 0, :]
    return reference + (states - reference.unsqueeze(-2)).mean(dim=-2)


@fused
def _combine(normalized, gate_logits, write_logits, read_offset, lambda_read,
             write_offset, lambda_write, read_index):
    """Both gates, the corrected read and the write weights, from the routing logits.

    Eager, this is a sigmoid, a broadcast multiply-add, a second broadcast multiply
    and a reduction, each reading and writing a full `[batch, tokens, branches,
    hidden]` tensor -- four kernels and three 84 MB intermediates at batch 2 x 4096.
    Fused, none of the intermediates is written.

    The one-hot read is taken directly rather than as a weighted sum, so at
    initialisation `correction` is exactly zero and the value is exactly branch
    `read_index` of the normalised stream.
    """
    read_gate = torch.sigmoid(gate_logits).unflatten(-1, normalized.shape[-2:])
    correction = read_offset.unsqueeze(-1) + lambda_read * read_gate
    value = normalized[..., read_index, :] + (correction * normalized).sum(-2)
    weights = (1 + write_offset) + lambda_write * torch.sigmoid(write_logits)
    return value, weights


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
        # One branch at a time keeps the temporaries [B,T,d]-sized rather than
        # [B,T,n,d]; Windows has no expandable_segments, so a backward that churns big
        # odd-sized blocks strands gigabytes of reserved-but-unallocated pool. The
        # branch views are already contiguous in the normalized dimension, and the
        # per-branch gain rides along with the norm, so the [B,T,n,d] stack is written
        # once instead of being read back and multiplied whole.
        normalized = torch.stack(
            [branch_norm(branch, norm, gain) for branch, gain
             in zip(states.unbind(-2), self.branch_gain_delta)], dim=-2)
        flattened = normalized.flatten(-2)
        # Linear is homogeneous, so 1/n applies to the narrow output instead of a
        # second [B,T,n,d] copy of the input. n is a power of two in practice and the
        # scaling is exact; at initialisation both gates are multiplied by zero anyway.
        scale = 1 / self.num_branches
        gate_logits = self.W_up(F.silu(self.W_down(flattened) * scale))
        write_logits = self.W_write(flattened) * scale
        value, weights = _combine(
            normalized, gate_logits, write_logits, self.read_offset, self.lambda_read,
            self.write_offset, self.lambda_write, self.read_index)
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
