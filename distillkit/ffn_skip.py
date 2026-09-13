# Copyright 2025 Arcee AI & DistillKit Contributors
"""Backward-compatibility shim for FFN skip.

Moved to ``distillkit.experimental.ffn_skip``.
"""

from __future__ import annotations

from distillkit.experimental.ffn_skip import (
    FFNAttenuate,
    FFNSkip,
    FFNSubstitute,
    attenuate_ffn,
    capture_ffn,
    estimate_savings,
    mlp_flops_per_token,
    model_flops_per_token,
    skip_ffn,
    substitute_ffn,
)

__all__ = [
    "FFNAttenuate",
    "FFNSkip",
    "FFNSubstitute",
    "attenuate_ffn",
    "capture_ffn",
    "estimate_savings",
    "mlp_flops_per_token",
    "model_flops_per_token",
    "skip_ffn",
    "substitute_ffn",
]
