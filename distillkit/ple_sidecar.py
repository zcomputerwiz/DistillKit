"""Flash-Next's PLE integration, ported to the student.

The 5M pilot said the sidecar helps early and then washes out, and the diagnostic was
that ``W_side_proj`` trained (0 -> 4.007) while the gated residual did not
(``gate_1_mean`` 0.5001 -> 0.5013). The n-gram features were being learned; nothing was
learning *when* to use them. This is the upstream answer to that question, transcribed.

Ported from ``transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextPLELayer``
(Qwen3.8-Flash-Next, ``qwen4_exp_text``), whose configuration this student already
mirrors: ``ple_layer_ids: [2]`` one-indexed is our ``sidecar_layer_index: 1``,
``ngram_size: 3`` and ``heads_per_ngram: 8`` give ``(3-1)*8 = 16`` n-gram heads of
``2560/16 = 160`` dimensions, which is exactly the ``sidecar_num_heads: 16`` x
``sidecar_head_dim: 160`` table this fork already reads.

**What we had, and why its gate could not learn.** Ours added the projected features
into the stream ungated and then applied a multi-branch ``GatedResidual`` to the sum::

    hidden = hidden + W_side_proj(features)
    hidden = gated_residual(hidden)

The gate is free parameters that must *discover* selectivity, and by that module's own
documented design it receives no gradient until the branches leave zero. It never did
discover it.

**What upstream does instead.** The gate is not a parameter at all -- it is a dot product
between a query read off the residual stream and a key read off the n-gram embedding::

    key   = norm_key(key_proj(features))
    value = value_proj(features)
    query = norm_query(hidden_states)
    gate  = (key * query).sum(-1) / sqrt(hidden_size)      # per position
    gate  = sign(gate) * sqrt(|gate|)                       # signed-sqrt compression
    out   = sigmoid(gate) * value
    out   = out + silu(depthwise_conv(norm_conv(out)))      # dilated, kernel 4
    hidden_states = hidden_states + out

So selectivity exists from the first step the value is nonzero: whether a token's n-gram
entry is used depends on whether it agrees with what the stream already carries. The
depthwise convolution is dilated by ``ngram_size`` so each output position sees the
same phase of neighbouring n-grams rather than blurring across them.

**The one deliberate divergence.** Upstream carries ``hc_count`` hyper-connection
streams and emits ``hc_count * hidden_size``; this student has no hyper-connections, so
``hc_count`` is 1. That collapses upstream's ``unflatten(-1, (hc_count, hidden_size))``
and the grouped RMSNorm to ordinary full-width operations, which is why they do not
appear below. Everything else -- the signed-sqrt, the sigmoid, the dilation, the residual
around the convolution, the ordering of the norms -- is transcribed rather than
reinterpreted.

**The norms, and why they are not ``nn.RMSNorm``.** Upstream stores a zero-initialised
weight and scales by ``(1 + w)`` computed in fp32; ``nn.RMSNorm`` stores a
ones-initialised weight and scales by ``w`` directly. At initialisation these are
bit-identical and their gradients match exactly, which is what made the substitution look
free. It is not. **bfloat16 spacing near 1.0 is 0.0078**, so a scale stored directly
cannot represent a learned deviation smaller than about 0.004 -- at this fork's 1e-4
learning rate every update to those weights would round away and the norms would sit
frozen at exactly 1.0. Storing the deviation instead puts it near zero, where bf16
spacing is ~1e-40 and the same update survives intact:

    delta 1e-3  ->  scale w    1.000000 (lost)      scale 1+w    1.000999 (kept)
    delta 1e-4  ->  scale w    1.000000 (lost)      scale 1+w    1.000100 (kept)

That is the identical failure mode this port exists to fix -- a parameter that cannot
move -- so ``_PLERMSNorm`` reproduces upstream's parameterisation rather than
substituting torch's. It also happens to make weight decay pull the scale toward 1 rather
than toward 0, though this fork decays only ``ndim >= 2`` so that never applied.

``value_proj`` is zero-initialised so the module is exactly the identity at load, which
matters when retrofitting onto a frozen, already-trained backbone. That does mean the
gate sees no gradient on the very first step, exactly as the old design did -- but unlike
the old design it does not have to *learn* selectivity afterwards, only to sharpen a
criterion it already computes.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["PLESidecar"]


class _PLERMSNorm(nn.Module):
    """Upstream's ``Qwen4ExpTextRMSNorm`` at ``group_size == dim``.

    Stores the *deviation* from unit scale and normalises in fp32, both of which matter
    in bf16 -- see the module docstring. Equivalent to ``nn.RMSNorm`` in fp32, not in
    bf16 once the weight has moved.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        # from_pretrained re-initialises parameters missing from the checkpoint, and its
        # default for a norm-shaped weight is ones -- which under (1 + w) is a scale of
        # 2.0, not 1.0. Mark it so _SidecarWeightInit restores the zero.
        self._sidecar_weight_init = "zero"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * (1.0 + self.weight.float())).type_as(x)


class PLESidecar(nn.Module):
    """``hidden -> hidden + conv(gate(query, key) * value)`` over n-gram features.

    ``features`` are the dequantized n-gram rows, ``[batch, seq, ngram_heads * head_dim]``
    -- upstream's ``ple_embedding(input_ids)`` output, which this fork reads from the
    frozen IQ4_NL table instead of hashing at run time.
    """

    def __init__(
        self,
        hidden_size: int,
        feature_dim: int,
        *,
        conv_kernel_size: int = 4,
        ngram_size: int = 3,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        if hidden_size <= 0 or feature_dim <= 0:
            raise ValueError("hidden_size and feature_dim must be positive")
        if conv_kernel_size < 1 or ngram_size < 1:
            raise ValueError("conv_kernel_size and ngram_size must be positive")
        self.hidden_size = hidden_size
        self.feature_dim = feature_dim
        # Upstream dilates the short convolution by ngram_size so a position sees the
        # same phase of adjacent n-grams; the state it needs is (K-1)*dilation wide.
        self.conv_dilation = ngram_size
        self.short_conv_state_len = (conv_kernel_size - 1) * ngram_size

        self.key_proj = nn.Linear(feature_dim, hidden_size, bias=False)
        self.value_proj = nn.Linear(feature_dim, hidden_size, bias=False)
        self.norm_key = _PLERMSNorm(hidden_size, eps=rms_norm_eps)
        self.norm_query = _PLERMSNorm(hidden_size, eps=rms_norm_eps)
        self.norm_conv = _PLERMSNorm(hidden_size, eps=rms_norm_eps)
        self.conv1d = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=conv_kernel_size,
            groups=hidden_size, dilation=self.conv_dilation, bias=False,
        )
        # Exactly the identity at initialisation, and -- unlike upstream -- *smoothly*
        # so. Zeroing value_proj alone makes the identity a knife-edge: norm_conv
        # renormalises whatever it receives to unit RMS, so the convolution branch
        # contributes at full scale the instant value_proj leaves zero, regardless of how
        # small it is. Measured on a stream of RMS 1.01, a value_proj at std 1e-4 already
        # injected 0.64 and at 1e-2 injected 1.84 -- larger than the stream it is meant to
        # nudge. Upstream can afford that because it trains the whole model jointly and
        # the backbone co-adapts; a retrofit onto a frozen backbone cannot, and the first
        # run of this module confirmed it (eval_loss 2.157 after two steps against the
        # backbone's own 0.5262). Zeroing the convolution too makes silu(conv(.)) exactly
        # zero, so the output starts as gated_value alone and both branches grow from
        # nothing.
        for zeroed in (self.value_proj, self.conv1d):
            nn.init.zeros_(zeroed.weight)
            zeroed._sidecar_weight_init = "zero"

    def _short_conv(self, gated: torch.Tensor) -> torch.Tensor:
        """Causal dilated depthwise convolution, left-padded so position t sees only <= t."""
        gated = gated.transpose(1, 2)
        gated = F.pad(gated, (self.short_conv_state_len, 0))
        gated = F.silu(self.conv1d(gated))
        return gated.transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        features = features.to(dtype=hidden_states.dtype)
        key_normed = self.norm_key(self.key_proj(features))
        value = self.value_proj(features)
        query_normed = self.norm_query(hidden_states)

        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        # Signed square root: compresses the dot product's range without losing its sign,
        # so a strongly disagreeing n-gram is suppressed rather than merely unamplified.
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        # Back to the stream's dtype before it touches the value. Under autocast the
        # reduction above is promoted to fp32 -- `sum` is on autocast's fp32 list -- and
        # multiplying a bf16 value by an fp32 gate promotes the whole value path, which
        # then propagates out of this module and through the rest of the model until
        # something meets a bf16 weight. Computing the gate itself in fp32 is wanted, a
        # 2560-wide dot product deserves it; carrying that width outward is not.
        gated_value = torch.sigmoid(gate).to(value.dtype) * value

        output = gated_value + self._short_conv(self.norm_conv(gated_value))
        return hidden_states + output

    @torch.no_grad()
    def gate_report(self, hidden_states: torch.Tensor, features: torch.Tensor) -> dict:
        """Gate statistics, for the same reason the old module reported them: if the gate
        does not move off its initial distribution the architecture is inert."""
        features = features.to(dtype=hidden_states.dtype)
        key_normed = self.norm_key(self.key_proj(features))
        query_normed = self.norm_query(hidden_states)
        raw = (key_normed * query_normed).sum(dim=-1) / math.sqrt(self.hidden_size)
        gate = torch.sigmoid(raw.abs().clamp_min(1e-6).sqrt() * raw.sign()).float()
        return {
            "ple/gate_mean": gate.mean().item(),
            "ple/gate_std": gate.std().item(),
            "ple/gate_frac_open": (gate > 0.6).float().mean().item(),
            "ple/gate_frac_shut": (gate < 0.4).float().mean().item(),
            "ple/value_norm": self.value_proj.weight.norm().item(),
            "ple/key_norm": self.key_proj.weight.norm().item(),
        }
