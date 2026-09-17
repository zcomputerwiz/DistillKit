# Copyright 2025 Arcee AI & DistillKit Contributors
"""Model architectures, adapters, and registry for DistillKit."""

from distillkit.models.registry import (
    register_student_model,
    resolve_student_class,
)
from distillkit.models.loader import is_flash_attn_available, load_student_model
from distillkit.models.qwen35.sidecar import Qwen35SidecarForCausalLM
from distillkit.models.qwen35.widened import Qwen35WidenedForCausalLM

__all__ = [
    "Qwen35SidecarForCausalLM",
    "Qwen35WidenedForCausalLM",
    "is_flash_attn_available",
    "register_student_model",
    "resolve_student_class",
    "load_student_model",
]
