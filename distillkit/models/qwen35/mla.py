"""Decoupled Multi-head Latent Attention for the Flash-Next full-attention layers.

DeepSeek's MLA replaces per-head key and value projections with one compressed latent
``c`` per token plus a single rotary key shared by every head. The cache then holds
``d_c + rope_dim`` numbers per token instead of ``num_key_value_heads * head_dim * 2``.

**Built directly rather than converted.** The published recipe extracts ``W_DKV``,
``W_UK`` and ``W_UV`` from a trained model by activation-aware SVD and repairs the
approximation with billions of tokens of distillation. That machinery exists to preserve a
pretrained model's function. Training from scratch needs none of it: ``d_c`` becomes a free
hyperparameter instead of a rank fitted to somebody else's weights, and there is no
reconstruction error to gate on.

**The rotary split is free here.** ``apply_rotary_pos_emb`` in this model rotates the first
``cos.shape[-1]`` dimensions of each head and passes the rest through, and
``partial_rotary_factor`` 0.25 at ``head_dim`` 64 makes that 16 rotated and 48 passed. So
laying each key out as ``[shared rotary 16 | content 48]`` gives the decoupled structure
with the stock rotary function and no reimplementation: the first slice is
position-dependent and shared, the rest is per-head content read out of the latent.

**What it costs and saves at this size.** Against the GQA it replaces -- 2 key/value heads
of 64, so 256 cached numbers per token per layer -- a latent of 128 plus 16 rotary caches
144, a 1.8x reduction. It adds parameters rather than removing them, because the
up-projections are per-head where GQA shared two heads' worth: 188,416 against 131,072 per
layer. A latent wider than 256 would cache *more* than the GQA did, which is the trap in
porting ``d_c`` from a model whose ``d_model`` is fourteen times larger.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5RMSNorm, apply_rotary_pos_emb, eager_attention_forward)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

__all__ = ["Qwen35LatentAttention", "mla_dimensions"]


def mla_dimensions(config) -> tuple[int, int, int]:
    """``(latent, rope_dim, content_dim)`` for a config, validated against the head."""
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    rope = getattr(config, "rope_parameters", {}) or {}
    factor = rope.get("partial_rotary_factor", 1.0)
    rope_dim = int(head_dim * factor)
    content_dim = head_dim - rope_dim
    if content_dim <= 0:
        raise ValueError("partial_rotary_factor %r leaves no content dimensions" % factor)
    latent = int(getattr(config, "mla_latent_dim", 128))
    if latent < 1:
        raise ValueError("mla_latent_dim must be positive; got %d" % latent)
    return latent, rope_dim, content_dim


class Qwen35LatentAttention(nn.Module):
    """Drop-in for ``Qwen3_5Attention`` with a compressed key/value latent.

    The query path is left exactly as the stock module has it, gate included, so a
    comparison against the GQA baseline isolates the key/value side. Every head gets its
    own key and value read out of the shared latent, so there is no grouping and
    ``num_key_value_groups`` is 1.
    """

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim",
                                config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = 1
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.latent, self.rope_dim, self.content_dim = mla_dimensions(config)
        bias = config.attention_bias

        # Query, unchanged from stock: the second half is Qwen's output gate.
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim * 2,
                                bias=bias)
        # One projection produces the compressed latent and the shared rotary key.
        self.kv_a_proj = nn.Linear(config.hidden_size, self.latent + self.rope_dim,
                                   bias=bias)
        self.kv_a_norm = Qwen3_5RMSNorm(self.latent, eps=config.rms_norm_eps)
        # Up-projection to per-head content keys and full values.
        self.kv_b_proj = nn.Linear(
            self.latent, self.num_heads * (self.content_dim + self.head_dim), bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size,
                                bias=bias)

        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # DeepSeek normalizes the latent and nothing after it: `kv_a_norm` is the only norm
        # on the key/value path, and the up-projection's output goes to attention as it is.
        # This fork previously added a content-half key norm, which neither parent has --
        # Qwen norms the whole head, DeepSeek norms the latent, and a norm over the content
        # half alone is a third thing. It is kept only to load checkpoints trained with it,
        # because it changes the function rather than the layout.
        self.content_key_norm = bool(getattr(config, "mla_content_key_norm", False))
        self.k_norm = (Qwen3_5RMSNorm(self.content_dim, eps=config.rms_norm_eps)
                       if self.content_key_norm else None)

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2,
            dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)

        compressed = self.kv_a_proj(hidden_states)
        latent, rotary_key = torch.split(compressed, [self.latent, self.rope_dim], dim=-1)
        latent = self.kv_a_norm(latent)

        # Rotating the shared slice on its own is equivalent to rotating the assembled
        # key -- `apply_rotary_pos_emb` touches exactly the first `rope_dim` dimensions
        # and passes the content half through -- and it is what lets the cache hold the
        # compressed form: the slice goes in already carrying its position, so replaying
        # it later needs no position bookkeeping.
        cos, sin = position_embeddings
        query_states, rotary_key = apply_rotary_pos_emb(
            query_states, rotary_key.unsqueeze(1), cos, sin)

        if past_key_values is not None:
            # The cache holds the latent and the rotary key, not the per-head keys and
            # values those expand into: `latent + rope_dim` numbers per token against
            # `2 * num_heads * head_dim`. The up-projection below then runs over the whole
            # cached sequence each step, which is the trade MLA makes and the reason
            # DeepSeek folds the up-projection into the query at serving time.
            latent, rotary_key = past_key_values.update(
                latent.unsqueeze(1), rotary_key, self.layer_idx)
            latent = latent.squeeze(1)

        kv_shape = latent.shape[:-1]
        projected = self.kv_b_proj(latent).view(
            *kv_shape, self.num_heads, self.content_dim + self.head_dim)
        content_key, value_states = torch.split(
            projected, [self.content_dim, self.head_dim], dim=-1)
        if self.k_norm is not None:
            content_key = self.k_norm(content_key)

        # One rotary key for every head: shared, and the only position-dependent part.
        shared = rotary_key.expand(*kv_shape[:1], self.num_heads, kv_shape[1],
                                   self.rope_dim)
        key_states = torch.cat([shared, content_key.transpose(1, 2)], dim=-1)
        value_states = value_states.transpose(1, 2)

        interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward)
        attn_output, attn_weights = interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, **kwargs)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output), attn_weights

    def cached_numbers_per_token(self) -> int:
        """What one token costs in the serving cache, for comparison with GQA.

        This is what the cache actually holds, not what it would hold in principle: the
        forward writes the latent and the rotary key, so handing `update` the expanded
        per-head keys -- `2 * num_heads * head_dim`, four times the GQA it replaces --
        would make this number a fiction.
        """
        return self.latent + self.rope_dim
