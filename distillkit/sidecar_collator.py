# Copyright 2025 Arcee AI & DistillKit Contributors
"""Backward-compatibility shim for sidecar data collator.

Moved to ``distillkit.experimental.sidecar_collator``.
"""

from __future__ import annotations

from distillkit.experimental.sidecar_collator import (
    SidecarDataCollator,
)

__all__ = [
    "SidecarDataCollator",
]
