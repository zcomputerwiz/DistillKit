# Copyright 2025 Arcee AI & DistillKit Contributors
"""Backward-compatibility shim for hyper connection.

Moved to ``distillkit.experimental.hyper_connection``.
"""

from __future__ import annotations

from distillkit.experimental.hyper_connection import (
    HyperConnection,
    HyperConnectionWarmupCallback,
    _GatedMean,
)

__all__ = [
    "HyperConnection",
    "HyperConnectionWarmupCallback",
    "_GatedMean",
]
