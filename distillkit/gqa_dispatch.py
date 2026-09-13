# Copyright 2025 Arcee AI & DistillKit Contributors
"""Backward-compatibility shim for grouped-query attention dispatch.

Moved to ``distillkit.models.qwen35.gqa_dispatch``.
"""

from __future__ import annotations

from distillkit.models.qwen35.gqa_dispatch import (
    fused_kernel_supports_gqa,
    install_expanded_gqa_attention,
)

__all__ = [
    "install_expanded_gqa_attention",
    "fused_kernel_supports_gqa",
]
