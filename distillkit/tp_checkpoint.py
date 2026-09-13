"""Backward compatibility shim for distillkit.parallel.checkpoint."""

from distillkit.parallel.checkpoint import (
    MARKER,
    TensorSpec,
    consolidated_state_dict,
    load_checkpoint,
    tensor_specs,
    write_layout,
)

__all__ = [
    "MARKER",
    "TensorSpec",
    "consolidated_state_dict",
    "load_checkpoint",
    "tensor_specs",
    "write_layout",
]
