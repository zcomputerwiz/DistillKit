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

from distillkit.experimental.widened_residual import _BranchNorm

#: The one fixed perturbation for the asymmetric recipient initialization. Mean-zero, so
#: the average branch gain -- and therefore the initial read -- is unchanged in exact
#: arithmetic. An engineering choice, not an optimum, and deliberately not swept.
DEFAULT_EPSILON = torch.tensor([-3.0, -1.0, 1.0, 3.0]) / 128.0


def _at_least_fp32(dtype):
    """``dtype`` unless it is narrower than fp32, in which case fp32."""
    return torch.promote_types(dtype, torch.float32)


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


class _GatedProjectMean(torch.autograd.Function):
    """`_GatedMean` with the up-projection folded in, so its output is never written.

    The gate logits are `W_up(code)`, one `[batch, tokens, branches * hidden]` tensor --
    268 MB at batch 64 by 1024 -- produced only to be split per branch, consumed
    immediately, and then held until backward. Projecting one branch at a time inside the
    loop never forms it: each `[batch, tokens, hidden]` slice is used and dropped, and
    backward recomputes the slice it needs from `code`, which is `lowrank` wide rather
    than `branches * hidden`. Across twenty sublayers that is the difference between
    holding five gigabytes of gate logits and holding none.

    The arithmetic is the same arithmetic. Forward accumulates the bf16 product in fp32,
    as `_GatedMean` does; backward rounds the logit gradient to the parameter dtype
    before its matmuls, which is what autograd did to it on the way into `W_up`.
    """

    @staticmethod
    def forward(ctx, normalized, code, weight):
        ctx.save_for_backward(normalized, code, weight)
        compute = _at_least_fp32(normalized.dtype)
        width = normalized.shape[-1]
        result = torch.zeros_like(normalized[..., 0, :], dtype=compute)
        for index, x in enumerate(normalized.unbind(-2)):
            rows = weight[index * width:(index + 1) * width]
            result.add_(x * F.linear(code, rows).sigmoid())
        return (result / normalized.shape[-2]).to(normalized.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        normalized, code, weight = ctx.saved_tensors
        compute = _at_least_fp32(normalized.dtype)
        branches, width = normalized.shape[-2], normalized.shape[-1]
        grad = grad_output.to(compute) / branches
        flat_code = code.flatten(0, -2)

        grad_normalized = torch.empty_like(normalized)
        grad_code = torch.zeros_like(code, dtype=compute)
        grad_weight = torch.empty_like(weight)
        for index in range(branches):
            rows = weight[index * width:(index + 1) * width]
            gate = F.linear(code, rows).sigmoid().to(compute)
            grad_normalized[..., index, :] = (grad * gate).to(normalized.dtype)
            # d/d(logit) of `x * sigmoid(logit)`, rounded to the parameter dtype exactly
            # where autograd would have rounded it.
            grad_logit = (grad * normalized[..., index, :] * gate * (1 - gate)
                          ).to(weight.dtype)
            grad_code += F.linear(grad_logit, rows.t()).to(compute)
            grad_weight[index * width:(index + 1) * width] = (
                grad_logit.flatten(0, -2).t() @ flat_code)
        return grad_normalized, grad_code.to(code.dtype), grad_weight


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
    #: How the branch read normalises, cheapest last. Measured at micro-batch 32 over
    #: `[32, 1024, 4, 512]`, forward and backward together:
    #:
    #: * ``exact`` -- 13.134 ms. Bit-identical to ``Qwen3_5RMSNorm``, which
    #:   ``recipient_initialize`` needs in order to reproduce a donor.
    #: * ``fast`` -- 11.952 ms. ``vector_norm`` for the variance instead of two fp32
    #:   copies of the stream. About 1e-7 relative, four orders below bf16's own.
    #: * ``fused`` -- 8.416 ms. ``F.rms_norm``, which normalises in the input dtype and
    #:   so rounds once more before the gain: about 5e-3 relative, roughly one bf16 ulp
    #:   and a real loss of precision rather than a rounding detail.
    NORM_MODES = ("exact", "fast", "fused")

    def __init__(self, hidden_size, num_branches=4, lowrank=320, layer_idx=0,
                 blend=0.0, norm_eps=1e-6, learnable_blend=False,
                 norm_mode="exact"):
        super().__init__()
        if min(hidden_size, num_branches, lowrank) < 1 or norm_eps <= 0:
            raise ValueError("routing dimensions and norm_eps must be positive")
        self.hidden_size, self.num_branches = hidden_size, num_branches
        self.read_index = layer_idx % num_branches
        self.norm_eps = norm_eps
        self.initial_blend = float(blend)
        # Exact by default: `recipient_initialize` reproduces a pretrained sublayer
        # bitwise through this route, so anything cheaper is available only to a model
        # with no donor to reproduce.
        if norm_mode not in self.NORM_MODES:
            raise ValueError("unknown norm_mode %r; expected one of %s"
                             % (norm_mode, list(self.NORM_MODES)))
        self.norm_mode = norm_mode
        self.learnable_blend = bool(learnable_blend)
        # fp32 either way. bfloat16 spacing at 0.10 is 2.44e-4 against an AdamW step of
        # roughly `lr`, so a bf16 blend at a useful initialisation is bit-frozen exactly
        # the way `sharpness` was; see ple_gated_sidecar.py and PROGRESS.md 2026-09-10.
        value = torch.tensor(float(blend), dtype=torch.float32)
        if self.learnable_blend:
            # A learned blend stays a device tensor and stays in the graph, so the CPU
            # pin and the constant-folded endpoints in `read` are both off for it. The
            # state_dict key is the same either way, so checkpoints cross between modes.
            self.blend = nn.Parameter(value)
        else:
            self.register_buffer("blend", value)
        self.set_blend(blend)
        # Extraction stores GGUF norm scale minus one, independently of student.
        self.branch_gain_delta = nn.Parameter(torch.zeros(num_branches, hidden_size))
        self.W_down = nn.Linear(num_branches * hidden_size, lowrank, bias=False)
        self.W_up = nn.Linear(lowrank, num_branches * hidden_size, bias=False)
        self.W_write = nn.Linear(num_branches * hidden_size, num_branches, bias=False)
        # Provenance, not state: the converted weights are what a checkpoint carries, and
        # these are restored from the model config on reload rather than by re-running
        # the initialization over trained weights. See `widened.recipient_initialize`.
        self.recipient_initialized = False
        self.recipient_mode = None

    @torch.no_grad()
    def recipient_initialize(self, norm, asymmetric=False, seed=0, epsilon=None):
        """Set this route to reproduce ``norm``'s sublayer with GR fully active.

        The recipient computes ``h + F(RMSNorm(h; gamma, eps))``. With ``W_up = 0`` the
        gate logits are exactly zero, so every read gate is ``1/2``; with ``W_write = 0``
        every write multiplier is ``2 * sigmoid(0) = 1``. Setting each branch gain to
        ``2 * gamma`` then makes the read

            mean_i(1/2 * RMSNorm(h; 2 gamma)) = RMSNorm(h; gamma)

        and gives every branch the recipient's own update, so branches that start equal
        stay equal and the mean collapse returns ``h``.

        Two details are easy to get wrong and both would be silent:

        ``Qwen3_5RMSNorm`` stores a *deviation* -- it applies ``1 + weight`` -- and this
        module also applies ``1 + branch_gain_delta``, as a *replacement* for the block
        norm rather than a gain on top of it. The effective gain we need is ``2 * gamma``
        with ``gamma = 1 + weight``, so the stored delta is ``1 + 2 * weight``. Writing
        ``2 * weight`` would be wrong by exactly one on every branch.

        ``W_down`` is seeded nonzero on purpose. Zeroing both factors of the read
        bottleneck would leave it with no gradient path even after ``W_up`` moves -- the
        same zero-multiplier trap that made the first borrowed-routing transfer inert.

        The asymmetric variant scales the gains by ``1 + epsilon_i`` with mean-zero
        ``epsilon``, which leaves the average read unchanged in exact arithmetic while
        letting the branches carry different gains. It is a predeclared comparison, not
        the primary candidate: the synthetic diagnostic in ``scratch/gr_retrofit`` found
        it advances symmetry onset by one or two optimizer steps and is then overtaken by
        the symmetric mode.
        """
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

        if not isinstance(norm, Qwen3_5RMSNorm):
            raise TypeError(
                "recipient initialization supports Qwen3_5RMSNorm, which applies "
                "1 + weight; got %s. Refusing rather than guessing the gain convention."
                % type(norm).__name__)
        if abs(float(norm.eps) - float(self.norm_eps)) > 0:
            raise ValueError("norm epsilon %r does not match the route's %r"
                             % (norm.eps, self.norm_eps))
        # Everything is built on the *recipient norm's* device. A norm that is already
        # on CUDA against branch scales built on the CPU is not a promotion case -- the
        # scales are one-dimensional, so the multiply raises rather than broadcasting --
        # and conversion has to work both before and after the model is placed.
        device = norm.weight.device
        gamma = 1.0 + norm.weight.detach().to(torch.float32)
        scale = torch.ones(self.num_branches, dtype=torch.float32, device=device)
        if asymmetric:
            values = DEFAULT_EPSILON if epsilon is None else torch.as_tensor(epsilon)
            values = values.to(device=device, dtype=torch.float32)
            if values.numel() != self.num_branches:
                raise ValueError("epsilon has %d entries for %d branches"
                                 % (values.numel(), self.num_branches))
            if float(values.sum().abs()) > 1e-6:
                raise ValueError("epsilon must be mean-zero to preserve the initial read")
            scale = 1.0 + values
        effective = 2.0 * gamma.unsqueeze(0) * scale.unsqueeze(-1)
        # Stored at fp32 or better, never at the route's current dtype: a recipient that
        # has already been cast to bf16 would otherwise round `1 + 2 * weight` on the
        # way in, and bf16 spacing near 3 is 1.6e-2 against an AdamW step near the
        # learning rate, so the gains would arrive rounded and then sit bit-frozen.
        self.branch_gain_delta.data = (effective - 1.0).to(
            _at_least_fp32(self.branch_gain_delta.dtype)
        ).to(self.branch_gain_delta.device)
        self.W_up.weight.zero_()
        self.W_write.weight.zero_()
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        fan_in = self.W_down.weight.shape[1]
        self.W_down.weight.data = (
            torch.randn(self.W_down.weight.shape, generator=generator,
                        dtype=torch.float32) / fan_in ** 0.5
        ).to(self.W_down.weight.dtype).to(self.W_down.weight.device)
        self.set_blend(1.0)
        self.recipient_initialized = True
        self.recipient_mode = "asymmetric" if asymmetric else "symmetric"
        return {"mode": self.recipient_mode, "seed": int(seed),
                "epsilon": scale.sub(1.0).tolist() if asymmetric else None}

    def _apply(self, fn, recurse=True):
        blend, gains = self.blend, self.branch_gain_delta.data
        result = super()._apply(fn, recurse)
        # fp32 always. A *scheduled* blend also goes to the CPU, because `read` consumes
        # it as a Python scalar and nothing else ever does, and a CUDA scalar there
        # costs a device synchronisation on each of this model's 64 sublayers -- 2.35 ms
        # per forward, doubled by gradient checkpointing's recompute -- to fetch a
        # number the CPU itself wrote. A *learned* blend has to sit where the arithmetic
        # is, so it follows the module like any other parameter.
        if self.learnable_blend:
            self.blend.data = self.blend.data.to(dtype=torch.float32)
        else:
            self.blend = blend.to(device="cpu", dtype=torch.float32)
        # The branch gains are pinned the same way, for the same reason one step up:
        # `read` promotes them to fp32 to use them, so bf16 storage buys nothing in the
        # forward and costs the whole gain update in the backward -- bf16 spacing near
        # the converted value of 3 is 1.6e-2 against an AdamW step near the learning
        # rate. A model-wide cast to a *wider* dtype is still honoured; this raises the
        # floor rather than pinning the gains to fp32 exactly. The pre-cast values are
        # restored rather than the post-cast ones, because undoing a narrowing cast
        # recovers the dtype but not the digits it threw away.
        applied = self.branch_gain_delta.data
        target = _at_least_fp32(applied.dtype)
        if applied.dtype != target:
            source = gains if gains.dtype == _at_least_fp32(gains.dtype) else applied
            self.branch_gain_delta.data = source.to(device=applied.device, dtype=target)
        return result

    @torch.no_grad()
    def set_blend(self, value):
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("hyper-connection blend must be finite and in [0, 1]")
        self.blend.fill_(value)

    def read(self, states, norm):
        if self.learnable_blend:
            return self._read_learned(states, norm)
        alpha = float(self.blend)
        if alpha == 0:
            # Skip donor arithmetic entirely: even poisoned donor weights cannot
            # spoil identity via 0*NaN. Use the original norm's exact operations.
            return norm(states[..., self.read_index, :]).contiguous(), None
        donor, weights = self._route(states)
        if alpha == 1:
            return donor.contiguous(), weights
        original = norm(states[..., self.read_index, :])
        return (original + alpha * (donor - original)).contiguous(), 1 + alpha * (weights - 1)

    def _route(self, states):
        """The donor read and the write weights, from one pass over the stream.

        `W_down` and `W_write` are both bias-free and both read the same flattened
        stream, and both are narrow on the output side -- `lowrank` and `num_branches`
        against `num_branches * hidden` on the input. So each spends nearly all its time
        reading the same `[batch, tokens, branches * hidden]` tensor, which is the
        largest thing in this function. One concatenated weight reads it once.

        The two modules stay, so checkpoints keep their own `W_down.weight` and
        `W_write.weight` and nothing about the parameterization changes; only the
        multiply is shared.
        """
        normalized = self._normalize(states)
        flattened = normalized.flatten(-2)
        projected = F.linear(flattened, torch.cat(
            [self.W_down.weight, self.W_write.weight], dim=0))
        low, write = projected.split(
            [self.W_down.out_features, self.num_branches], dim=-1)
        code = F.silu(low / self.num_branches)
        donor = _GatedProjectMean.apply(normalized, code, self.W_up.weight)
        return donor, 2 * torch.sigmoid(write / self.num_branches)

    def _normalize(self, states):
        """Every branch normalized and gained, in one kernel over `[..., n, d]`.

        RMSNorm reduces over the last dimension, so the branch axis is just more leading
        shape, and a `[branches, hidden]` gain broadcasts against it exactly as a
        `[hidden]` gain broadcasts against one branch. Per-branch calls needed a stack to
        put the results back together, which is a full-width copy per sublayer on top of
        one kernel launch per branch.

        This is what `_BranchNorm.backward` reducing over `grad.ndim - gain.ndim` is for:
        against the old `grad.ndim - 1` the branch axis would be summed away and every
        branch would receive the same gain gradient.
        """
        gain = 1.0 + self.branch_gain_delta.to(
            torch.promote_types(self.branch_gain_delta.dtype, torch.float32))
        if self.norm_mode == "fused":
            # One fused kernel over the stream, then the gain. `F.rms_norm` takes a
            # weight of the normalised shape and this gain is per branch, so it cannot
            # ride along inside the kernel -- the extra elementwise pass is still well
            # ahead of the hand-written form.
            return (F.rms_norm(states, (states.shape[-1],), None, self.norm_eps)
                    * gain).to(states.dtype)
        return _BranchNorm.apply(states, gain, self.norm_eps,
                                 self.norm_mode == "fast")

    def _donor(self, states, norm):
        """Donor read, donor write weights, and the student's own block input."""
        donor, weights = self._route(states)
        return donor, weights, norm(states[..., self.read_index, :])

    def _read_learned(self, states, norm):
        """The same interpolation with the blend kept in the graph.

        No constant-folded endpoint here. Skipping the donor at blend == 0 would skip
        its gradient too, and the parameter could never leave zero -- the same
        zero-multiplier trap that made the first borrowed-routing transfer inert, in a
        form that would read as "the blend decided it wanted nothing".

        The derivative is well behaved at both ends: d(x_a)/d(alpha) is
        `donor - student`, which does not vanish at alpha = 0, so the parameter can
        move off any starting value. It is deliberately unbounded -- where it settles
        is the measurement, and clamping to [0, 1] would pin it at a boundary with zero
        gradient and no way back.
        """
        donor, weights, original = self._donor(states, norm)
        alpha = self.blend.to(donor.dtype)
        return ((original + alpha * (donor - original)).contiguous(),
                1 + alpha * (weights - 1))

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
                f"{prefix}/blend_learnable": float(self.learnable_blend),
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
