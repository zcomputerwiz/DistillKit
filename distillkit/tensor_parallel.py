"""Backward compatibility shim for distillkit.parallel.collectives."""

from distillkit.parallel.collectives import (
    AllReduce,
    Reduce,
    Replicate,
    all_reduce,
    peer_capable,
    reduce_to,
    replicate,
)

__all__ = [
    "AllReduce",
    "Reduce",
    "Replicate",
    "all_reduce",
    "peer_capable",
    "reduce_to",
    "replicate",
]
