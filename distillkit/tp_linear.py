"""Backward compatibility shim for distillkit.parallel.linear."""

from distillkit.parallel.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    split_sizes,
)

__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "split_sizes",
]
