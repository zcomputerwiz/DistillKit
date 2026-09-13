# Copyright 2025 Arcee AI & DistillKit Contributors
"""Core engine optimization and memory primitives for DistillKit.

This package provides model-agnostic utilities for scaling LLM training and distillation:
- anchor_tap: Selective hidden-state capture via forward hooks.
- chunked_head: Folded lm_head projection inside the loss loop to avoid full logits materialization.
- chunked_ce: Memory-cheap causal LM cross-entropy and Accelerate bf16 output preservation.
- frozen_prefix: Autograd-free prefix execution for staged/adapter training.
- sortish_sampler: Microbatch-level length grouping with batch shuffling.
"""

from distillkit.core.anchor_tap import (
    AnchorTap,
    CapturedStates,
    anchor_module,
)
from distillkit.core.chunked_ce import (
    DEFAULT_CHUNK_BYTES,
    chunk_tokens_for,
    chunked_causal_lm_loss,
    keep_bf16_forward_outputs,
    maybe_install_chunked_loss,
)
from distillkit.core.chunked_head import (
    HeadContext,
    chunked_head_loss,
    head_device,
)
from distillkit.core.frozen_prefix import (
    no_grad_prefix,
)
from distillkit.core.sortish_sampler import (
    DEFAULT_SORT_WINDOW,
    SortishSampler,
    sortish_indices,
)

__all__ = [
    # Anchor Tap
    "AnchorTap",
    "CapturedStates",
    "anchor_module",
    # Chunked Head
    "HeadContext",
    "chunked_head_loss",
    "head_device",
    # Chunked Cross-Entropy
    "DEFAULT_CHUNK_BYTES",
    "chunk_tokens_for",
    "chunked_causal_lm_loss",
    "keep_bf16_forward_outputs",
    "maybe_install_chunked_loss",
    # Frozen Prefix
    "no_grad_prefix",
    # Sortish Sampler
    "DEFAULT_SORT_WINDOW",
    "SortishSampler",
    "sortish_indices",
]
