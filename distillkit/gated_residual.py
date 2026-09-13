# Copyright 2025 Arcee AI & DistillKit Contributors
"""Backward-compatibility shim for gated residual.

Moved to ``distillkit.experimental.gated_residual``.
"""

from __future__ import annotations

from distillkit.experimental.gated_residual import (
    GateReport,
    GatedResidual,
)

__all__ = [
    "GateReport",
    "GatedResidual",
]
