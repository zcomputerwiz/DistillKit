"""Backward compatibility shim for distillkit.parallel.collectives."""

from distillkit.parallel.collectives import (
    AllReduce,
    Reduce,
    Replicate,
    _sum_to_each,
    all_reduce,
    peer_capable,
    reduce_to,
    replicate,
)

__all__ = [
    "AllReduce",
    "_sum_to_each",
    "Reduce",
    "Replicate",
    "all_reduce",
    "peer_capable",
    "reduce_to",
    "replicate",
]
