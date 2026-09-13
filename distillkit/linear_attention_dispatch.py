# Copyright 2025 Arcee AI & DistillKit Contributors
"""Backward-compatibility shim for Qwen 3.5 linear attention dispatch.

Moved to ``distillkit.models.qwen35.linear_attention_dispatch``.
"""

from __future__ import annotations

from distillkit.models.qwen35.linear_attention_dispatch import (
    fused_linear_attention_available,
    install_device_aware_linear_attention,
)

__all__ = [
    "install_device_aware_linear_attention",
    "fused_linear_attention_available",
]
