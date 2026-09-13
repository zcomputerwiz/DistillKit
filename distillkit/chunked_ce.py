"""Backward compatibility shim for distillkit.core.chunked_ce."""

from distillkit.core.chunked_ce import (
    DEFAULT_CHUNK_BYTES,
    chunk_tokens_for,
    chunked_causal_lm_loss,
    keep_bf16_forward_outputs,
    maybe_install_chunked_loss,
)

__all__ = [
    "chunked_causal_lm_loss",
    "DEFAULT_CHUNK_BYTES",
    "chunk_tokens_for",
    "maybe_install_chunked_loss",
    "keep_bf16_forward_outputs",
]
