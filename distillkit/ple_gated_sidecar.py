# Copyright 2025 Arcee AI & DistillKit Contributors
"""Backward-compatibility shim for direction-gated PLE sidecar.

Moved to ``distillkit.experimental.ple_gated_sidecar``.
"""

from __future__ import annotations

from distillkit.experimental.ple_gated_sidecar import (
    DirectionGatedPLESidecar,
)

__all__ = [
    "DirectionGatedPLESidecar",
]
