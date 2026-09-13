# Copyright 2025 Arcee AI & DistillKit Contributors
"""Backward-compatibility shim for transcribed PLE sidecar.

Moved to ``distillkit.experimental.ple_sidecar``.
"""

from __future__ import annotations

from distillkit.experimental.ple_sidecar import (
    PLESidecar,
    _PLERMSNorm,
)

__all__ = [
    "PLESidecar",
    "_PLERMSNorm",
]
