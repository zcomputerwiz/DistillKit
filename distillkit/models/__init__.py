"""Explicit model classes for DistillKit architecture extensions."""

from .qwen35_sidecar import Qwen35SidecarForCausalLM
from .qwen35_widened import Qwen35WidenedForCausalLM

__all__ = ["Qwen35SidecarForCausalLM", "Qwen35WidenedForCausalLM"]
