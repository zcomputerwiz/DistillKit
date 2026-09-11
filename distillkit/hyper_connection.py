"""Flash-Next Gated Residual routing, with an explicit student-to-donor bridge.

Equations 30--34: https://arxiv.org/html/2608.30320v1
Reference: transformers Qwen4ExpTextGatedResidual (6b07e4510e3f).
The donor norm replaces the block norm; it is not a gain on top of it.
The student retains its own final norm/mean collapse, not the donor's final mixer.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn import functional as F
from transformers import TrainerCallback

from distillkit.widened_residual import _BranchNorm


class _GatedMean(torch.autograd.Function):
    """Branchwise gate/product, without full widened gate/product temporaries.

Save normalized states and logits, recompute narrow gates in backward. The BF16
product is rounded before FP32 accumulation, as in the reference's BF16 mean.
Backward uses FP32 intermediate arithmetic, like our existing branch norm.
"""
    @staticmethod
    def forward(ctx, normalized, logits):
        ctx.save_for_backward(normalized, logits)
        compute = torch.promote_types(normalized.dtype, torch.float32)
        result = torch.zeros_like(normalized[..., 0, :], dtype=compute)
        for x, z in zip(normalized.unbind(-2), logits.unbind(-2)):
            # add_ promotes the bf16 product on the fly. The explicit .to(compute)
            # this replaces allocated a second full-width fp32 branch per iteration;
            # bf16 -> fp32 is exact either way, so the result is bit-identical.
            result.add_(x * z.sigmoid())
        return (result / normalized.shape[-2]).to(normalized.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        normalized, logits = ctx.saved_tensors
        compute = torch.promote_types(normalized.dtype, torch.float32)
        grad = grad_output.to(compute) / normalized.shape[-2]
        dx, dz = torch.empty_like(normalized), torch.empty_like(logits)
        for i in range(normalized.shape[-2]):
            gate = logits[..., i, :].sigmoid().to(compute)
            dx[..., i, :] = grad * gate
            dz[..., i, :] = grad * normalized[..., i, :] * gate * (1 - gate)
        return dx, dz


class HyperConnection(nn.Module):
    """Donor endpoint at blend=1; exact original pre-norm route at blend=0.

Intermediate reads interpolate *block inputs*, including their separate norms:
    x_a = x_student + a * (x_donor - x_student)
    s_a = 1 + a * (s_donor - 1)
    R'_i = R_i + s_a[i] * F(x_a)
No extra anchor, static gate offset, or pretrained block gain at the donor end.
Blend is a scheduled FP32 buffer, not an AdamW parameter. Keeping it at zero
deliberately makes the donor inert; use the warmup callback to activate it.
"""
    def __init__(self, hidden_size, num_branches=4, lowrank=320, layer_idx=0,
                 blend=0.0, norm_eps=1e-6):
        super().__init__()
        if min(hidden_size, num_branches, lowrank) < 1 or norm_eps <= 0:
            raise ValueError("routing dimensions and norm_eps must be positive")
        self.hidden_size, self.num_branches = hidden_size, num_branches
        self.read_index = layer_idx % num_branches
        self.norm_eps = norm_eps
        self.initial_blend = float(blend)
        self.register_buffer("blend", torch.tensor(float(blend), dtype=torch.float32))
        self.set_blend(blend)
        # Extraction stores GGUF norm scale minus one, independently of student.
        self.branch_gain_delta = nn.Parameter(torch.zeros(num_branches, hidden_size))
        self.W_down = nn.Linear(num_branches * hidden_size, lowrank, bias=False)
        self.W_up = nn.Linear(lowrank, num_branches * hidden_size, bias=False)
        self.W_write = nn.Linear(num_branches * hidden_size, num_branches, bias=False)

    def _apply(self, fn, recurse=True):
        blend = self.blend
        result = super()._apply(fn, recurse)
        # fp32 so a scheduled 0.015 survives a bf16 cast, and on the CPU because
        # `read` consumes it as a Python scalar and nothing else ever does. A CUDA
        # scalar here costs a device synchronisation on every one of this model's 64
        # sublayers -- 2.35 ms per forward, doubled by gradient checkpointing's
        # recompute -- to fetch a number the CPU itself wrote. Keeping it host-side
        # removes the stall without a shadow copy that could drift from the buffer
        # on a load path that does not go through `set_blend`.
        self.blend = blend.to(device="cpu", dtype=torch.float32)
        return result

    @torch.no_grad()
    def set_blend(self, value):
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("hyper-connection blend must be finite and in [0, 1]")
        self.blend.fill_(value)

    def read(self, states, norm):
        alpha = float(self.blend)
        if alpha == 0:
            # Skip donor arithmetic entirely: even poisoned donor weights cannot
            # spoil identity via 0*NaN. Use the original norm's exact operations.
            return norm(states[..., self.read_index, :]).contiguous(), None
        normalized = torch.stack([
            _BranchNorm.apply(branch, 1.0 + gain.to(torch.promote_types(
                gain.dtype, torch.float32)), self.norm_eps)
            for branch, gain in zip(states.unbind(-2), self.branch_gain_delta)
        ], dim=-2)
        flattened = normalized.flatten(-2)
        logits = self.W_up(F.silu(self.W_down(flattened) / self.num_branches))
        donor = _GatedMean.apply(normalized, logits.unflatten(
            -1, (self.num_branches, self.hidden_size)))
        weights = 2 * torch.sigmoid(self.W_write(flattened) / self.num_branches)
        if alpha == 1:
            return donor.contiguous(), weights
        original = norm(states[..., self.read_index, :])
        return (original + alpha * (donor - original)).contiguous(), 1 + alpha * (weights - 1)

    def write(self, states, output, weights):
        if weights is None:
            return states + output.unsqueeze(-2)
        # Branchwise separate mul/add matches the donor's rounding (addcmul fuses).
        # Only the returned widened stream is materialized, not the product.
        return torch.stack([branch + weight.unsqueeze(-1) * output
                            for branch, weight in zip(states.unbind(-2), weights.unbind(-1))], dim=-2)

    @torch.no_grad()
    def gate_report(self, prefix="residual_stream"):
        return {f"{prefix}/blend": float(self.blend),
                f"{prefix}/donor_norm_delta": self.branch_gain_delta.float().norm().item()}


class HyperConnectionWarmupCallback(TrainerCallback):
    """Linear schedule indexed by completed optimizer steps, including resumes.

Set only outside forward/backward so checkpoint replay sees the same coefficient.
After step N the model (and any checkpoint/evaluation) has alpha(N).
"""
    def __init__(self, start=0.0, target=1.0, steps=0):
        if steps < 0 or not all(math.isfinite(v) and 0 <= v <= 1 for v in (start, target)):
            raise ValueError("invalid hyper-connection warmup")
        self.start, self.target, self.steps = start, target, steps

    def _set(self, model, step):
        fraction = min(max(step, 0) / self.steps, 1) if self.steps else 0
        value = self.start + fraction * (self.target - self.start)
        for module in model.modules():
            if isinstance(module, HyperConnection):
                module.set_blend(value)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._set(model, state.global_step)
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        self._set(model, state.global_step)
        return control
