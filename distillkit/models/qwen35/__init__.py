# Copyright 2025 Arcee AI & DistillKit Contributors
"""Qwen 3.5 specific model adapters, converters, and parallel modules."""

from distillkit.models.qwen35.tp_gated_delta import (
    ConvChannelPlan,
    conv_channel_plan,
    head_group_plan,
)
from distillkit.models.qwen35.tp_gated_delta_module import (
    TensorParallelGatedDeltaNet,
)

__all__ = [
    "ConvChannelPlan",
    "conv_channel_plan",
    "head_group_plan",
    "TensorParallelGatedDeltaNet",
]
