# Copyright 2025 Arcee AI & DistillKit Contributors
"""Backward-compatibility shim for native PLE sidecar.

Moved to ``distillkit.experimental.native_ple``.
"""

from __future__ import annotations

from distillkit.experimental.native_ple import (
    NativePLESidecar,
    _chunked_norm,
    native_hash_config,
)

__all__ = [
    "NativePLESidecar",
    "_chunked_norm",
    "native_hash_config",
]
