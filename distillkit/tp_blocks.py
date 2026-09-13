"""Backward compatibility shim for distillkit.parallel.blocks."""

from distillkit.parallel.blocks import (
    TensorParallelAttention,
    TensorParallelMLP,
    _assert_head_major,
    shard_decoder_layer,
)

__all__ = [
    "TensorParallelAttention",
    "TensorParallelMLP",
    "_assert_head_major",
    "shard_decoder_layer",
]
