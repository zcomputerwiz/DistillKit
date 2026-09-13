"""Backward compatibility shim for distillkit.parallel.model and distillkit.parallel.sync."""

from distillkit.parallel.model import (
    _shard_tied_embeddings,
    shard_model,
    sharded_parameter_report,
)
from distillkit.parallel.sync import (
    sync_replicated_gradients,
)

__all__ = [
    "_shard_tied_embeddings",
    "shard_model",
    "sharded_parameter_report",
    "sync_replicated_gradients",
]
