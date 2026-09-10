"""PLE with per-stream direction gates, sized for a widened residual rather than for
upstream's four hyper-connection streams.

This is what the measurements in ``PROGRESS.md`` (2026-09-10) leave standing of
``ple_sidecar.PLESidecar``, which is a faithful transcription of upstream's layer. Three
results shaped it, and each removes something:

**``key_proj`` does not earn its 26.2M multiplies.** It is 80% of the layer's arithmetic
and produces one scalar per stream. On upstream's own trained weights the keys come out
of ``norm_key(key_proj(features))`` nearly collinear -- mean pairwise cosine 0.81 to 0.96
-- so there is nothing for a query to match against, and substituting the *mean* key for
every position reproduces 65-93% of the gate. The gate is admission control read off the
residual stream, not a relevance match. So the key path is replaced by a learned
direction per stream: ``gate_s = f(query_s . g_s)``, which is what the trained module
computes anyway, at 2,560x fewer multiplies and without a 26.2M-parameter tensor.

**One direction per stream is enough.** A controlled ablation at matched gate scale
(``scratch/table_capacity_probe.py``) put one direction at -0.003626 [-0.004397,
-0.002896] against an ungated linear read and four directions at -0.003614 [-0.004370,
-0.002895] -- indistinguishable, with value norms matching to 0.1%. The apparent
advantage of four in the first run was scale: summing k gates starts the value path at an
effective multiplier of k/2, and ``(sum_i g_i) W f = (mean_i g_i)(k W) f`` when ``W`` is
free. ``gate_directions`` therefore defaults to 2 rather than 1 only because that
ablation could not test what *this* module is for: with one residual stream every gate
reads the same vector, so it could not distinguish directions from streams. Here the
streams differ, and each carries its own directions.

**The gate must be learned.** Four *frozen* random directions bought -0.000061
[-0.000163, +0.000035] -- an interval spanning zero. Whatever the gate contributes, it is
a criterion the run discovers, not any non-linear function of the stream.

**The convolution branch never sees the gate.** ``norm_conv`` RMS-normalises
``gate * value`` and RMS normalisation cancels a positive scalar, so its input is
``norm_conv(value)`` however the gate moves -- exact as ``eps -> 0``, and departing as
``eps / (gate^2 * mean(value^2))`` (checked in ``scratch/ple_restructure_check.py``). The
per-stream norm weight then folds into the depthwise filters, since scaling channel c of
a depthwise convolution's input is the same as scaling that channel's filter. Both are
applied here: one reduction over the shared value feeds every stream's convolution, and
``norm_conv`` does not exist as a parameter.

``norm_query``'s weight is gone for the same kind of reason. Upstream scales the
normalised query elementwise by ``(1 + w_s)`` and then dots it with the key; since
``(q * (1 + w_s)) . g = q . ((1 + w_s) * g)``, a per-stream norm weight is exactly
absorbable into that stream's direction, and keeping both would be two parameterisations
of one function.

**Identity at load** is preserved the way ``PLESidecar`` establishes it: ``value_proj``
and ``conv1d`` are both zero, so the gated value and the convolution branch are both
exactly zero and the module returns its input unchanged. The gate directions are
*not* zero, and must not be -- the signed square root has ``sign(0) = 0`` and its
``clamp_min`` flattens ``abs`` at the origin, so a zero direction receives exactly zero
gradient and stays at 0.5 for the entire run. Measured: ``|grad|`` 0.0 at ``g = 0``
against 4.85 at ``g ~ N(0, 0.02)``. Because the value starts at zero the gate cannot
break the identity regardless of where its directions point.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["DirectionGatedPLESidecar"]


class DirectionGatedPLESidecar(nn.Module):
    """``stream[..., hc, d] -> stream + gate_s * value + silu(conv_s(rms(value)))``.

    ``features`` are the dequantized n-gram rows, ``[batch, seq, ngram_heads * head_dim]``.
    The value is shared across streams, as upstream's is; only admission differs.
    """

    def __init__(
        self,
        hidden_size: int,
        feature_dim: int,
        *,
        hc_count: int = 2,
        gate_directions: int = 2,
        conv_kernel_size: int = 4,
        ngram_size: int = 3,
        rms_norm_eps: float = 1e-6,
        gate_init_std: float = 0.02,
    ):
        super().__init__()
        if hidden_size <= 0 or feature_dim <= 0:
            raise ValueError("hidden_size and feature_dim must be positive")
        if conv_kernel_size < 1 or ngram_size < 1:
            raise ValueError("conv_kernel_size and ngram_size must be positive")
        if hc_count < 1 or gate_directions < 1:
            raise ValueError("hc_count and gate_directions must be positive")
        if gate_init_std <= 0:
            raise ValueError("gate_init_std must be positive; a zero direction cannot train")
        self.hidden_size = hidden_size
        self.feature_dim = feature_dim
        self.hc_count = hc_count
        self.gate_directions = gate_directions
        self.eps = rms_norm_eps
        self._last_gate_stats: torch.Tensor | None = None
        # Upstream dilates the short convolution by ngram_size so a position sees the
        # same phase of adjacent n-grams; the state it needs is (K-1)*dilation wide.
        self.conv_dilation = ngram_size
        self.short_conv_state_len = (conv_kernel_size - 1) * ngram_size

        self.value_proj = nn.Linear(feature_dim, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.empty(hc_count, gate_directions, hidden_size))
        self.conv1d = nn.Conv1d(
            hc_count * hidden_size, hc_count * hidden_size, kernel_size=conv_kernel_size,
            groups=hc_count * hidden_size, dilation=self.conv_dilation, bias=False,
        )
        # Held on the module, not the parameter: `from_pretrained` re-initialises
        # missing tensors after materialising them and its hook is called per module, so
        # a mark on a bare Parameter is never seen and the directions would come back as
        # whatever HF's default is -- including, for a checkpoint saved before this field
        # existed, zero. A zero direction never trains.
        self.gate_init_std = gate_init_std
        nn.init.normal_(self.gate, std=gate_init_std)
        for zeroed in (self.value_proj, self.conv1d):
            nn.init.zeros_(zeroed.weight)
            zeroed._sidecar_weight_init = "zero"

    def _admission(self, stream: torch.Tensor) -> torch.Tensor:
        """``2 * mean_k sigmoid(signed_sqrt(rms(stream_s) . g_sk / sqrt(d)))``.

        The mean rather than the sum, so that a change in ``gate_directions`` does not
        silently rescale the value path: every width starts at 1.0 and spans (0, 2).
        """
        query = stream.float()
        query = query * torch.rsqrt(query.pow(2).mean(-1, keepdim=True) + self.eps)
        raw = torch.einsum("...hd,hkd->...hk", query, self.gate.float())
        raw = raw / math.sqrt(self.hidden_size)
        # Signed square root: compresses the dot product's range without losing its sign,
        # so a strongly disagreeing stream suppresses the value rather than merely not
        # amplifying it. clamp_min keeps the derivative finite at the origin.
        gate = torch.sigmoid(raw.abs().clamp_min(1e-6).sqrt() * raw.sign())
        if self.training:
            with torch.no_grad():
                flat = gate.detach().float()
                self._last_gate_stats = torch.stack([
                    flat.mean(), flat.std(),
                    (flat > 0.6).float().mean(), (flat < 0.4).float().mean(),
                ])
        return 2.0 * gate.mean(-1, keepdim=True)

    def _short_conv(self, value: torch.Tensor) -> torch.Tensor:
        """Every stream's causal dilated convolution over one shared normalised value.

        The gate is absent on purpose: it cancels inside the normalisation it would have
        passed through, so this branch depends only on the value. One reduction serves
        every stream, and the per-stream norm weight lives in the filters.
        """
        normed = value.float()
        normed = normed * torch.rsqrt(normed.pow(2).mean(-1, keepdim=True) + self.eps)
        wide = normed.to(value.dtype).repeat(1, 1, self.hc_count).transpose(1, 2)
        wide = F.pad(wide, (self.short_conv_state_len, 0))
        wide = F.silu(self.conv1d(wide)).transpose(1, 2)
        return wide.unflatten(-1, (self.hc_count, self.hidden_size))

    def forward(self, stream: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if stream.shape[-2:] != (self.hc_count, self.hidden_size):
            raise ValueError(
                f"expected a stream ending in ({self.hc_count}, {self.hidden_size}), "
                f"got {tuple(stream.shape)}"
            )
        features = features.to(dtype=stream.dtype)
        value = self.value_proj(features)
        gate = self._admission(stream).to(value.dtype)
        return stream + gate * value.unsqueeze(-2) + self._short_conv(value)

    @torch.no_grad()
    def gate_report(self, prefix: str = "ple") -> dict:
        """Is this thing learning, and is its gate selecting rather than scaling?

        ``value_norm`` and ``conv_norm`` start at exactly zero, so any nonzero value is
        movement. ``gate_std`` is the one to watch: a computed gate cannot sit at its
        initialisation the way a learned scalar can, but it can still return nearly the
        same number for every token, which is a constant scale wearing a gate's clothes.
        """
        report = {
            f"{prefix}/value_norm": self.value_proj.weight.float().norm().item(),
            f"{prefix}/conv_norm": self.conv1d.weight.float().norm().item(),
            f"{prefix}/gate_direction_norm": self.gate.float().norm().item(),
        }
        for stream in range(self.hc_count):
            report[f"{prefix}/gate_direction_norm_{stream}"] = (
                self.gate[stream].float().norm().item())
        if self._last_gate_stats is not None:
            mean, std, open_, shut = self._last_gate_stats.tolist()
            report[f"{prefix}/gate_mean"] = mean
            report[f"{prefix}/gate_std"] = std
            report[f"{prefix}/gate_frac_open"] = open_
            report[f"{prefix}/gate_frac_shut"] = shut
        return report
