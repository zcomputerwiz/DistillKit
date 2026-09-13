"""Backward compatibility shim for distillkit.models.qwen35.tp_gated_delta_module and distillkit.parallel.sync."""

from distillkit.models.qwen35.tp_gated_delta_module import (
    TensorParallelGatedDeltaNet,
)
from distillkit.parallel.sync import (
    clip_grad_norm,
    replicated_parameter_groups,
    sync_replicated_gradients,
)

__all__ = [
    "TensorParallelGatedDeltaNet",
    "clip_grad_norm",
    "replicated_parameter_groups",
    "sync_replicated_gradients",
]
