# Copyright 2025 Arcee AI & DistillKit Contributors
"""Backward-compatibility shim for n-gram hashing.

Moved to ``distillkit.experimental.ngram_hash``.
"""

from __future__ import annotations

from distillkit.experimental.ngram_hash import (
    FLASH_NEXT_NGRAM_CONFIG,
    NGramHashConfig,
    NGramHasher,
    build_layer_multipliers,
    find_nth_prime_after,
    splitmix64,
)

__all__ = [
    "FLASH_NEXT_NGRAM_CONFIG",
    "NGramHashConfig",
    "NGramHasher",
    "build_layer_multipliers",
    "find_nth_prime_after",
    "splitmix64",
]
