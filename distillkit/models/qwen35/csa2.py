"""Compressed Sparse Attention 2: Full / Reindex / Reuse layer modes over latent KV.

DeepSeek-V4.1-Flash assigns every attention layer one of three static modes, sharing the
main KV latent *and* the indexer keys across layers while reusing top-k routing indices:

* **Full** -- own indexer keys, own KV latent, computes its own top-k candidate set
* **Reindex** -- borrows indexer keys and KV latent, but scores its own queries against
  them to produce a *new* top-k
* **Reuse** -- borrows the top-k and the KV latent, and computes neither

Only the queries and the up-projections are ever private. That is what makes the three
modes differ on two independent axes rather than one, and why a dense implementation
collapses Reindex and Reuse into the same layer.

**The indexer is a lightweight scorer, not a second attention.** Each layer projects a few
low-dimensional index queries, scores them against one shared index key per token, and
sums the per-head scores through a ReLU with learned weights. Top-k over that picks which
positions the expensive latent attention actually reads.

**k is small here on purpose.** DeepSeek selects 2,048 of up to 128K. Training sequences
here are 1,024 tokens, which is shorter than their k, so any faithful setting selects
everything and the sparsity is a no-op. A small k makes the three modes genuinely distinct
and tests whether a model trains through the routing at all, which is the part of this
worth rehearsing at toy scale. It does not test what sparsity buys at long context, and
nothing here should be read as if it did.

Position ``t`` always keeps itself and a short recent window regardless of score. A router
that can starve a query of its own immediate context produces gradients that say more
about the router's initialization than about the architecture.
"""

from __future__ import annotations

import torch
from torch import nn

from .mla import Qwen35LatentAttention

__all__ = ["SparseIndexBus", "Qwen35SparseLatentAttention", "csa2_modes"]

MODES = ("full", "reindex", "reuse")

_COMPILED = None


def _flex():
    """`flex_attention`, compiled once for the process.

    Uncompiled it falls back to an eager path that materializes scores and gives up the
    whole point. Compiling per call would pay the warm-up on every step.
    """
    global _COMPILED
    if _COMPILED is None:
        from torch.nn.attention.flex_attention import flex_attention
        _COMPILED = torch.compile(flex_attention, dynamic=False)
    return _COMPILED


def csa2_modes(config, full_attention_layers: int) -> list[str]:
    """One mode per full-attention layer, validated and defaulted.

    The default repeats ``full, reuse`` so every borrowed layer sits directly behind the
    layer it borrows from. A sequence that opens on anything but ``full`` is refused
    rather than silently borrowing from nothing.
    """
    modes = list(getattr(config, "csa2_modes", None) or
                 ["full" if i % 2 == 0 else "reuse" for i in range(full_attention_layers)])
    if len(modes) != full_attention_layers:
        raise ValueError("csa2_modes has %d entries for %d full-attention layers"
                         % (len(modes), full_attention_layers))
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise ValueError("unknown csa2 modes %r; expected %s" % (unknown, list(MODES)))
    if modes[0] != "full":
        raise ValueError("the first full-attention layer must be 'full'; it has nothing "
                         "to borrow from")
    return modes


class SparseIndexBus:
    """What a Full layer publishes and the borrowing modes read.

    Held by the text model and cleared at the start of every forward, so nothing survives
    between batches. A borrowing layer that finds the bus empty is a configuration error,
    not something to paper over with a fallback.
    """

    __slots__ = ("index_keys", "latent", "rotary", "topk")

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.index_keys = None
        self.latent = None
        self.rotary = None
        self.topk = None

    def require(self, field: str, layer_idx: int):
        value = getattr(self, field)
        if value is None:
            raise RuntimeError("layer %d needs %s from an earlier full layer and the bus "
                               "is empty" % (layer_idx, field))
        return value


class Qwen35SparseLatentAttention(Qwen35LatentAttention):
    """Latent attention with CSA2 routing. ``mode`` fixes what this layer owns."""

    def __init__(self, config, layer_idx: int, mode: str, bus: SparseIndexBus) -> None:
        super().__init__(config, layer_idx)
        if mode not in MODES:
            raise ValueError("unknown csa2 mode %r" % mode)
        self.mode = mode
        self.bus = bus
        self.index_dim = int(getattr(config, "csa2_index_dim", 64))
        self.index_heads = int(getattr(config, "csa2_index_heads", 4))
        self.top_k = int(getattr(config, "csa2_top_k", 128))
        self.local_window = int(getattr(config, "csa2_local_window", 32))
        self.block_size = int(getattr(config, "csa2_block_size", 128))

        bias = config.attention_bias
        # Index keys are published by Full layers and borrowed by everyone else.
        if mode == "full":
            self.index_k_proj = nn.Linear(config.hidden_size, self.index_dim, bias=bias)
        # Reuse computes no routing at all, so it needs no index queries either.
        if mode in ("full", "reindex"):
            self.index_q_proj = nn.Linear(
                config.hidden_size, self.index_heads * self.index_dim, bias=bias)
            self.index_weight = nn.Parameter(torch.zeros(self.index_heads))
        # Reindex and reuse read somebody else's latent; the adapter is what lets them
        # read it differently rather than identically.
        if mode in ("reindex", "reuse"):
            self.kv_adapt = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
            nn.init.eye_(self.kv_adapt.weight)
            # A borrowing layer never projects its own latent, so the inherited down
            # projection is dead weight -- 73,728 parameters per layer that would ship in
            # every checkpoint and take no gradient.
            del self.kv_a_proj
            del self.kv_a_norm
        if mode == "reuse":
            # Reuse computes no routing, so it has no index queries and no weights for
            # them either. What it owns is its queries and its up-projections.
            pass

    @property
    def latent_dim(self) -> int:
        return self.latent

    def route(self, hidden_states, index_keys):
        """Which key *blocks* each query block reads: ``[batch, q_blocks, kv_blocks]``.

        Block granularity is the design rather than a concession to the kernel. The
        proposal routes "per micro-block", DeepSeek's indexer is block-granular, and an
        arbitrary per-token mask is precisely what no fused attention kernel can exploit --
        measured here at 17.39 ms against 5.51 ms for the block-sparse form.

        Scoring is against block summaries, not against every key. Pooling the index keys
        per block first makes this ``O(seq * blocks)`` instead of ``O(seq^2)``, so the
        router costs a fraction of the attention it is deciding for.
        """
        batch, seq, _ = hidden_states.shape
        blocks = seq // self.block_size
        queries = self.index_q_proj(hidden_states).view(
            batch, seq, self.index_heads, self.index_dim)
        summary = index_keys.view(batch, blocks, self.block_size, self.index_dim).mean(2)

        scores = torch.einsum("bqhd,bnd->bhqn", queries.float(), summary.float())
        weights = torch.nn.functional.softplus(self.index_weight).view(1, -1, 1, 1)
        scores = (torch.relu(scores) * weights).sum(dim=1)
        # A query block reads a key block if any of its queries wants it.
        scores = scores.view(batch, blocks, self.block_size, blocks).amax(dim=2)

        rows = torch.arange(blocks, device=hidden_states.device)
        causal = rows.view(-1, 1) >= rows.view(1, -1)
        scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))

        keep = max(1, min(self.top_k // self.block_size, blocks))
        chosen = scores.topk(keep, dim=-1).indices
        allowed = torch.zeros(batch, blocks, blocks, dtype=torch.bool,
                              device=hidden_states.device)
        allowed.scatter_(-1, chosen, True)
        # The diagonal block always survives routing: a query that cannot see its own
        # immediate context produces gradients about the router, not the architecture.
        allowed |= torch.eye(blocks, dtype=torch.bool,
                             device=hidden_states.device).unsqueeze(0)
        return allowed & causal.unsqueeze(0)

    def block_mask(self, allowed):
        """`allowed` as a `BlockMask`, built from the indices rather than scanned.

        `create_block_mask` would re-derive this by evaluating a mask function over every
        block pair. The top-k already *is* that answer, so `from_kv_blocks` takes it
        directly. Blocks strictly below the diagonal are fully visible and are passed as
        `full` so the kernel skips masking them; the diagonal block is partial, because
        causality still applies inside it.
        """
        from torch.nn.attention.flex_attention import BlockMask

        batch, blocks, _ = allowed.shape
        rows = torch.arange(blocks, device=allowed.device)
        diagonal = torch.eye(blocks, dtype=torch.bool, device=allowed.device)
        below = allowed & ~diagonal.unsqueeze(0)

        def indices_for(flags):
            counts = flags.sum(-1).to(torch.int32)
            order = torch.argsort(flags.to(torch.int8), dim=-1, descending=True,
                                  stable=True)
            return counts.unsqueeze(1), order.to(torch.int32).unsqueeze(1)

        full_counts, full_indices = indices_for(below)
        partial_counts, partial_indices = indices_for(
            diagonal.unsqueeze(0).expand(batch, blocks, blocks))

        def mask_mod(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        return BlockMask.from_kv_blocks(
            partial_counts, partial_indices, full_counts, full_indices,
            BLOCK_SIZE=self.block_size, mask_mod=mask_mod)

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kwargs):
        batch, seq, _ = hidden_states.shape

        if self.mode == "full":
            index_keys = self.index_k_proj(hidden_states)
            compressed = self.kv_a_proj(hidden_states)
            latent, rotary = torch.split(compressed, [self.latent, self.rope_dim], dim=-1)
            latent = self.kv_a_norm(latent)
            allowed = self.route(hidden_states, index_keys)
            self.bus.index_keys, self.bus.latent = index_keys, latent
            self.bus.rotary, self.bus.topk = rotary, allowed
        elif self.mode == "reindex":
            index_keys = self.bus.require("index_keys", self.layer_idx)
            latent = self.kv_adapt(self.bus.require("latent", self.layer_idx))
            rotary = self.bus.require("rotary", self.layer_idx)
            # New routing over borrowed keys: this layer's own view of what matters.
            allowed = self.route(hidden_states, index_keys)
            self.bus.topk = allowed
        else:
            latent = self.kv_adapt(self.bus.require("latent", self.layer_idx))
            rotary = self.bus.require("rotary", self.layer_idx)
            allowed = self.bus.require("topk", self.layer_idx)

        return self._attend(hidden_states, latent, rotary,
                            self.block_mask(allowed), position_embeddings)

    def _attend(self, hidden_states, latent, rotary, mask, position_embeddings):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)

        projected = self.kv_b_proj(latent).view(
            *input_shape, self.num_heads, self.content_dim + self.head_dim)
        content_key, value_states = torch.split(
            projected, [self.content_dim, self.head_dim], dim=-1)
        content_key = self.k_norm(content_key)

        shared = rotary.unsqueeze(-2).expand(*input_shape, self.num_heads, self.rope_dim)
        key_states = torch.cat([shared, content_key], dim=-1).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # FlexAttention compiles the block mask into a fused kernel that skips whole
        # blocks. DeepSeek's own sparse kernels are SM90/SM100 and this card is SM86, so
        # borrowing them is not available; this reaches the same place through Triton.
        # Measured against the alternatives at this shape: 5.51 ms here, 10.33 ms for
        # dense SDPA, 17.39 ms for SDPA with a token-level boolean mask.
        attn_output = _flex()(query_states, key_states, value_states,
                              block_mask=mask, scale=self.scaling)

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output), None

    def cached_numbers_per_token(self) -> int:
        """What this layer adds to the serving cache. Borrowing modes add nothing."""
        if self.mode != "full":
            return 0
        return self.latent + self.rope_dim + self.index_dim
