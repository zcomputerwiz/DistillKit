# Copyright 2025 Arcee AI & DistillKit Contributors
"""Backward-compatibility shim for n-gram table.

Moved to ``distillkit.experimental.ngram_table``.
"""

from __future__ import annotations

from distillkit.experimental.ngram_table import (
    FLASH_NEXT_TABLE,
    GGUFNGramTable,
    IQ4NL_BLOCK,
    IQ4NL_KVALUES,
    IQ4NL_TYPE_SIZE,
    IQ4NLDequant,
    NGramTableSpec,
    _KVALUES_NP,
    dequantize_iq4nl_rows,
    read_gguf_ple_metadata,
)

__all__ = [
    "FLASH_NEXT_TABLE",
    "GGUFNGramTable",
    "IQ4NL_BLOCK",
    "IQ4NL_KVALUES",
    "IQ4NL_TYPE_SIZE",
    "IQ4NLDequant",
    "NGramTableSpec",
    "_KVALUES_NP",
    "dequantize_iq4nl_rows",
    "read_gguf_ple_metadata",
]
