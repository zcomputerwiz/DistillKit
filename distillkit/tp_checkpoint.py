"""Backward compatibility shim for distillkit.parallel.checkpoint."""

from distillkit.parallel.checkpoint import (
    MARKER,
    TensorSpec,
    consolidated_state_dict,
    load_checkpoint,
    load_consolidated_state_dict,
    tensor_specs,
    training_layout,
    write_layout,
)

__all__ = [
    "MARKER",
    "TensorSpec",
    "consolidated_state_dict",
    "load_checkpoint",
    "load_consolidated_state_dict",
    "tensor_specs",
    "training_layout",
    "write_layout",
]
