# Copyright 2025 Arcee AI & DistillKit Contributors
"""Qwen 3.5 specific model adapters, converters, runtime dispatches, and parallel modules."""

from distillkit.models.qwen35.convert_gguf import (
    convert_gguf_to_hf,
    derive_text_config,
    expected_key_set,
)
from distillkit.models.qwen35.gqa_dispatch import (
    fused_kernel_supports_gqa,
    install_expanded_gqa_attention,
)
from distillkit.models.qwen35.linear_attention_dispatch import (
    fused_linear_attention_available,
    install_device_aware_linear_attention,
)
from distillkit.models.qwen35.sidecar import (
    Qwen35SidecarForCausalLM,
)
from distillkit.models.qwen35.tp_gated_delta import (
    ConvChannelPlan,
    conv_channel_plan,
    head_group_plan,
)
from distillkit.models.qwen35.tp_gated_delta_module import (
    TensorParallelGatedDeltaNet,
)
from distillkit.models.qwen35.widened import (
    Qwen35WidenedForCausalLM,
)

__all__ = [
    # Architectures
    "Qwen35SidecarForCausalLM",
    "Qwen35WidenedForCausalLM",
    # Conversion
    "convert_gguf_to_hf",
    "derive_text_config",
    "expected_key_set",
    # Dispatches
    "install_device_aware_linear_attention",
    "fused_linear_attention_available",
    "install_expanded_gqa_attention",
    "fused_kernel_supports_gqa",
    # Tensor Parallelism
    "ConvChannelPlan",
    "conv_channel_plan",
    "head_group_plan",
    "TensorParallelGatedDeltaNet",
]
