"""Backward compatibility shim for distillkit.parallel.vocab."""

from distillkit.parallel.vocab import (
    Collect,
    VocabParallelEmbedding,
    VocabParallelHead,
    VocabShardedLogits,
    collect_to,
)

__all__ = [
    "Collect",
    "VocabParallelEmbedding",
    "VocabParallelHead",
    "VocabShardedLogits",
    "collect_to",
]
