# Copyright 2025 Arcee AI & DistillKit Contributors
"""Backward-compatibility shim for widened residual stream.

Moved to ``distillkit.experimental.widened_residual``.
"""

from __future__ import annotations

from distillkit.experimental.widened_residual import (
    WidenedResidual,
    _BranchNorm,
    _combine,
    branch_norm,
    collapse_residual,
    offload_stream_boundaries,
)

__all__ = [
    "WidenedResidual",
    "_BranchNorm",
    "_combine",
    "branch_norm",
    "collapse_residual",
    "offload_stream_boundaries",
]
