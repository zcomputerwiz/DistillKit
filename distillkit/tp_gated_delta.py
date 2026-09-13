"""Backward compatibility shim for distillkit.models.qwen35.tp_gated_delta."""

from distillkit.models.qwen35.tp_gated_delta import (
    ConvChannelPlan,
    conv_channel_plan,
    head_group_plan,
)

__all__ = [
    "ConvChannelPlan",
    "conv_channel_plan",
    "head_group_plan",
]
