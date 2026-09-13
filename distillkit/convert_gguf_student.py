# Copyright 2025 Arcee AI & DistillKit Contributors
"""Backward-compatibility shim for Qwen 3.5 GGUF conversion.

Moved to ``distillkit.models.qwen35.convert_gguf``.
"""

from __future__ import annotations

from distillkit.models.qwen35.convert_gguf import (
    EOS_TOKEN_ID,
    GGUF_ARCH,
    TOKENIZER_FILES,
    _COMMON_MAP,
    _Dims,
    _FULL_MAP,
    _LINEAR_MAP,
    _decode_tensor,
    _invert_ssm_a,
    _unreorder_v_heads,
    convert_gguf_to_hf,
    derive_text_config,
    expected_key_set,
    main,
    map_gguf_tensor,
    read_kv,
)

__all__ = [
    "EOS_TOKEN_ID",
    "GGUF_ARCH",
    "TOKENIZER_FILES",
    "_COMMON_MAP",
    "_Dims",
    "_FULL_MAP",
    "_LINEAR_MAP",
    "_decode_tensor",
    "_invert_ssm_a",
    "_unreorder_v_heads",
    "convert_gguf_to_hf",
    "derive_text_config",
    "expected_key_set",
    "main",
    "map_gguf_tensor",
    "read_kv",
]

if __name__ == "__main__":
    main()
