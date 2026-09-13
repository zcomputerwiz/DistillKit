# Copyright 2025 Arcee AI & DistillKit Contributors
"""Backward-compatibility shim for Qwen 3.5 sidecar model.

Moved to ``distillkit.models.qwen35.sidecar``.
"""

from __future__ import annotations

from distillkit.models.qwen35.sidecar import (
    Qwen35SidecarForCausalLM,
    _DirectionGatedNGramSidecar,
    _DonorReaderNGramSidecar,
    _NGramSidecar,
    _PLENGramSidecar,
    _SidecarDecoderLayer,
    _SidecarTextModel,
    _SidecarWeightInit,
    _build_sidecar,
    _set_sidecar_defaults,
)

__all__ = [
    "Qwen35SidecarForCausalLM",
    "_DirectionGatedNGramSidecar",
    "_DonorReaderNGramSidecar",
    "_NGramSidecar",
    "_PLENGramSidecar",
    "_SidecarDecoderLayer",
    "_SidecarTextModel",
    "_SidecarWeightInit",
    "_build_sidecar",
    "_set_sidecar_defaults",
]
