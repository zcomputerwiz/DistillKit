# Copyright 2025 Arcee AI & DistillKit Contributors
"""Backward-compatibility shim for Qwen 3.5 widened model.

Moved to ``distillkit.models.qwen35.widened``.
"""

from __future__ import annotations

from distillkit.models.qwen35.widened import (
    Qwen35WidenedForCausalLM,
    WidenedDecoderLayer,
    _WidenedTextModel,
    _WidenedWeightInit,
)

__all__ = [
    "Qwen35WidenedForCausalLM",
    "WidenedDecoderLayer",
    "_WidenedTextModel",
    "_WidenedWeightInit",
]
