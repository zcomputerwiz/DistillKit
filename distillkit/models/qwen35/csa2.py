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

**The router is trained, not merely consulted.** Top-k is discrete: indices, a scatter and
a boolean mask carry no gradient, so an indexer that only selected blocks would never
learn which blocks are worth selecting -- its parameters would end a backward pass with
``grad=None``, which is what this module did before ``router_columns`` existed. The
indexer's score is therefore also *added to the attention logits*, so a position the
indexer favours is read more strongly and the loss can say whether that was right. It is
added by appending the score's own factors to the query and the key rather than through a
``score_mod``, which puts it in the existing GEMM at no measurable cost; both sides are
normalized first, so the router can move a logit by at most ``index_gate``. DeepSeek
trains the indexer instead by distilling the dense attention distribution into it; that
needs the dense distribution, which is the thing sparsity exists to avoid computing.

**k is small here on purpose.** DeepSeek selects 2,048 of up to 128K. Training sequences
here are 1,024 tokens, which is shorter than their k, so any faithful setting selects
everything and the sparsity is a no-op. A small k makes the three modes genuinely distinct
and tests whether a model trains through the routing at all, which is the part of this
worth rehearsing at toy scale. It does not test what sparsity buys at long context, and
nothing here should be read as if it did.

Position ``t`` always keeps itself and enough preceding blocks to cover
``csa2_local_window`` tokens, regardless of score. A router that can starve a query of its
own immediate context produces gradients that say more about the router's initialization
than about the architecture.

**Routing is causal at token granularity, not only at block granularity.** One routing
decision serves a whole block of queries, so it may use only what the earliest query in
that block can see -- otherwise a token's history depends on tokens that follow it, and
the causal mask cannot undo that, because the mask constrains what is read and not what
chose it. Two things follow: the decision is scored from the block's leading query rather
than pooled over all of them, and only blocks lying wholly in the past compete, which
keeps the diagonal block's own key summary out of the decision.

**Training only.** Routing is defined over whole blocks of a sequence that is present all
at once, and nothing here writes to or reads from a KV cache, so incremental decoding is
refused rather than silently mis-routed. Padding is refused for the same reason: a block
is routed as a unit and has no way to represent half of it being absent.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .mla import Qwen35LatentAttention

__all__ = ["SparseIndexBus", "Qwen35SparseLatentAttention", "csa2_modes",
           "routing_report"]

MODES = ("full", "reindex", "reuse")

_COMPILED = None


def _flex():
    """`flex_attention`, compiled once for the process.

    Uncompiled it falls back to an eager path that materializes scores and gives up the
    whole point. Compiling per call would pay the warm-up on every step.
    """
    global _COMPILED
    if _COMPILED is None:
        try:
            from torch.nn.attention.flex_attention import flex_attention
        except ImportError as error:  # pragma: no cover - depends on the installed torch
            raise RuntimeError(
                "CSA2 needs FlexAttention and BlockMask.from_kv_blocks, which arrived in "
                "torch 2.5; this interpreter has torch %s. The package floor of 2.0 is "
                "what the rest of DistillKit needs, not what this module needs."
                % torch.__version__) from error
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


def routing_report(model):
    """Per-layer routing statistics for every CSA2 layer in a model, in layer order.

    Returns an empty list for a model without CSA2, so a caller can record it
    unconditionally. A run that reports only its loss cannot tell a router that learned
    to route from one that quietly collapsed onto the blocks it gets for free.
    """
    rows = []
    for module in model.modules():
        if not isinstance(module, Qwen35SparseLatentAttention):
            continue
        statistics = module.routing_statistics()
        if statistics is not None:
            rows.append(dict(layer=module.layer_idx, **statistics))
    return rows


class SparseIndexBus:
    """What a Full layer publishes and the borrowing modes read.

    Held by the text model -- not by the config, which is serialized and shared between
    models -- and cleared at the start of every forward, so nothing survives between
    batches. A borrowing layer that finds the bus empty is a configuration error, not
    something to paper over with a fallback.

    The bus is written during the forward pass and read in layer order, so anything that
    re-runs a layer's forward out of order sees the wrong publisher. Gradient
    checkpointing does exactly that, which is why the model refuses the combination
    instead of producing quietly wrong gradients.
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
    """Latent attention with CSA2 routing. ``mode`` fixes what this layer owns.

    ``bus`` may be left ``None`` at construction and injected afterwards; the text model
    builds its layers first and then hands every one of them the same bus.
    """

    def __init__(self, config, layer_idx: int, mode: str,
                 bus: SparseIndexBus | None = None) -> None:
        super().__init__(config, layer_idx)
        if mode not in MODES:
            raise ValueError("unknown csa2 mode %r" % mode)
        self.mode = mode
        self.bus = bus
        # The last forward's block selection, kept so a run can report whether its router
        # is still choosing anything. Bool at [batch, blocks, blocks] -- a few kilobytes,
        # detached, out of the graph.
        self.last_allowed = None
        self.index_dim = int(getattr(config, "csa2_index_dim", 64))
        self.index_heads = int(getattr(config, "csa2_index_heads", 4))
        self.top_k = int(getattr(config, "csa2_top_k", 128))
        self.local_window = int(getattr(config, "csa2_local_window", 32))
        self.block_size = int(getattr(config, "csa2_block_size", 128))
        if self.block_size < 1:
            raise ValueError("csa2_block_size must be positive; got %d" % self.block_size)
        if self.local_window < 0:
            raise ValueError("csa2_local_window must not be negative; got %d"
                             % self.local_window)

        bias = config.attention_bias
        # Index keys are published by Full layers and borrowed by everyone else.
        if mode == "full":
            self.index_k_proj = nn.Linear(config.hidden_size, self.index_dim, bias=bias)
        # Reuse computes no routing at all, so it needs no index queries either.
        if mode in ("full", "reindex"):
            self.index_q_proj = nn.Linear(
                config.hidden_size, self.index_heads * self.index_dim, bias=bias)
            self.index_weight = nn.Parameter(torch.zeros(self.index_heads))
            # How far the router may move an attention logit. This is the whole gradient
            # path into the indexer, so it starts at 1 rather than at 0: a gate of zero
            # would zero the gradient it exists to carry.
            self.index_gate = nn.Parameter(torch.ones(()))
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

    @property
    def latent_dim(self) -> int:
        return self.latent

    @property
    def local_blocks(self) -> int:
        """Preceding blocks forced open so ``local_window`` tokens are always visible.

        A query at offset 0 of block ``i`` wants tokens back to ``i * block - window``,
        which lands in block ``i - ceil(window / block)``. Forcing that many blocks makes
        the window claim true for every query rather than only for those far enough from
        a block boundary.
        """
        return -(-self.local_window // self.block_size)

    def _project(self, hidden_states):
        """Every projection that reads the stream, in one multiply.

        A Full layer runs four of them -- the query, the latent, the index keys and the
        index queries -- and three are narrow: 144, 64 and 256 outputs against the
        query's 1024, on a 512-wide input. Each therefore spends most of its time reading
        the same `[batch, tokens, hidden]` tensor, so reading it once and splitting the
        result is strictly less work.

        The modules stay and keep their own weights, so checkpoints and the
        parameterization are unchanged; only the multiply is shared. Concatenating the
        weights costs a copy of the weights, which are four orders of magnitude smaller
        than the activations they are multiplied against.
        """
        modules = [self.q_proj]
        if self.mode == "full":
            modules += [self.kv_a_proj, self.index_k_proj]
        if self.mode in ("full", "reindex"):
            modules.append(self.index_q_proj)
        if len(modules) == 1:
            return (self.q_proj(hidden_states),)
        weight = torch.cat([module.weight for module in modules], dim=0)
        bias = (torch.cat([module.bias for module in modules], dim=0)
                if modules[0].bias is not None else None)
        fused = torch.nn.functional.linear(hidden_states, weight, bias)
        return fused.split([module.out_features for module in modules], dim=-1)

    def route(self, hidden_states, index_keys, queries=None):
        """``(allowed, effective)``: the hard block selection, and the query behind it.

        ``allowed`` is ``[batch, q_blocks, kv_blocks]`` booleans -- what the kernel skips
        blocks by, computed under ``no_grad`` because top-k, a scatter and a boolean mask
        carry no gradient anyway. ``effective`` is the head-collapsed index query per
        token, which ``_attend`` folds into the attention logits; that fold is the only
        thing that gives the indexer a gradient at all.

        Block granularity is the design rather than a concession to the kernel. The
        proposal routes "per micro-block", DeepSeek's indexer is block-granular, and an
        arbitrary per-token mask is precisely what no fused attention kernel can exploit --
        measured here at 17.39 ms against 5.51 ms for the block-sparse form.

        Scoring is against block summaries, not against every key, and from one leading
        query per block rather than every query. That makes this ``O(blocks^2)`` instead
        of ``O(seq^2)``, so the router costs a fraction of the attention it is deciding
        for -- but the reason it takes the leading query is causality, not cost. See the
        note at the pooling step.
        """
        batch, seq, _ = hidden_states.shape
        blocks = seq // self.block_size
        if queries is None:
            queries = self.index_q_proj(hidden_states)
        queries = queries.view(batch, seq, self.index_heads, self.index_dim)
        weights = torch.nn.functional.softplus(self.index_weight)

        with torch.no_grad():
            summary = index_keys.view(
                batch, blocks, self.block_size, self.index_dim).mean(2)
            # One decision serves every token in the block, so it may only use what the
            # *earliest* of them can see. Pooling the block's queries -- an amax over all
            # of them, as this did -- lets the block's last token change what its first
            # token is allowed to read: measured at 18 of 200 random single-token
            # mutations, and 191 of 200 at larger ones. Taking the leading query instead
            # is the most informative choice that stays causal, because position
            # `i * block_size` precedes every other position in block `i`.
            leaders = queries.view(
                batch, blocks, self.block_size, self.index_heads, self.index_dim)[:, :, 0]
            scores = torch.einsum("bqhd,bnd->bhqn", leaders.float(), summary.float())
            scores = (torch.relu(scores) * weights.view(1, -1, 1, 1)).sum(dim=1)

            rows = torch.arange(blocks, device=hidden_states.device)
            offsets = rows.view(-1, 1) - rows.view(1, -1)
            # The recent blocks always survive routing: a query that cannot see its own
            # immediate context produces gradients about the router, not the
            # architecture.
            local = (offsets >= 0) & (offsets <= self.local_blocks)
            # Only blocks that are complete in the past compete. This keeps a top-k slot
            # from being spent on a block that is forced open anyway, and it keeps the
            # diagonal block's summary -- the one place a key pool holds tokens from the
            # query's own future -- out of the decision entirely.
            eligible = offsets > self.local_blocks
            scores = scores.masked_fill(~eligible.unsqueeze(0), float("-inf"))

            keep = max(1, min(self.top_k // self.block_size, blocks))
            chosen = scores.topk(keep, dim=-1).indices
            allowed = torch.zeros(batch, blocks, blocks, dtype=torch.bool,
                                  device=hidden_states.device)
            allowed.scatter_(-1, chosen, True)
            # An early block has no eligible candidates, so its top-k over an all -inf
            # row returns arbitrary indices; this drops them.
            allowed &= eligible.unsqueeze(0)
            allowed |= local.unsqueeze(0)
            allowed &= (offsets >= 0).unsqueeze(0)

        # The heads collapse here. Summing head scores through a ReLU is what makes them
        # distinct, and the ReLU cannot be folded into a dot product; the linear part can,
        # and sum_h w_h * (Q_h . K) is exactly (sum_h w_h Q_h) . K.
        effective = (queries * weights.view(1, 1, -1, 1)).sum(dim=2)
        return allowed, effective

    def block_mask(self, allowed):
        """`allowed` as a `BlockMask`, built from the indices rather than scanned.

        `create_block_mask` would re-derive this by evaluating a mask function over every
        block pair. The top-k already *is* that answer, so `from_kv_blocks` takes it
        directly. Blocks strictly below the diagonal are fully visible and are passed as
        `full` so the kernel skips masking them; the diagonal block is partial, because
        causality still applies inside it.

        This is correct only because query and key blocks are the same size, the sequence
        divides evenly into them, and causality is the sole token-level constraint. The
        forward pass enforces all three rather than trusting them.
        """
        from torch.nn.attention.flex_attention import BlockMask

        batch, blocks, _ = allowed.shape
        rows = torch.arange(blocks, device=allowed.device)
        # A block above the diagonal passed as `full` is attended with no mask evaluated
        # at all, so the kernel would read the future outright. `route` already returns a
        # causal `allowed`; this is here because the class is easy to drive directly and
        # that mistake is silent.
        allowed = allowed & (rows.view(-1, 1) >= rows.view(1, -1)).unsqueeze(0)
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

    def _check_shapes(self, seq, past_key_values):
        """Refuse the shapes this routing cannot represent, rather than mis-routing them.

        Both are silent otherwise: a sequence that does not divide drops its tail block,
        and a one-token decode step computes zero blocks and picks nothing at all. Padding
        is refused one level up, where the raw mask is still visible.
        """
        if past_key_values is not None:
            raise RuntimeError(
                "CSA2 is training-only: routing is defined over blocks of a sequence "
                "that is present at once, and no layer writes the KV cache. Run with "
                "use_cache=False; incremental decoding needs a compressed cache and a "
                "decode-time router that do not exist yet.")
        if seq < self.block_size or seq % self.block_size:
            raise ValueError(
                "CSA2 needs a sequence length that is a positive multiple of "
                "csa2_block_size %d; got %d. Pad the batch to a multiple before the "
                "model, not inside it -- padding is not routable here."
                % (self.block_size, seq))

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kwargs):
        batch, seq, _ = hidden_states.shape
        self._check_shapes(seq, past_key_values)
        if self.bus is None:
            raise RuntimeError(
                "layer %d has no SparseIndexBus; the text model injects one after it "
                "builds its layers" % self.layer_idx)

        if self.mode == "full":
            query, compressed, index_keys, queries = self._project(hidden_states)
            latent, rotary = torch.split(compressed, [self.latent, self.rope_dim], dim=-1)
            latent = self.kv_a_norm(latent)
            allowed, effective = self.route(hidden_states, index_keys, queries)
            self.bus.index_keys, self.bus.latent = index_keys, latent
            self.bus.rotary, self.bus.topk = rotary, allowed
        elif self.mode == "reindex":
            query, queries = self._project(hidden_states)
            index_keys = self.bus.require("index_keys", self.layer_idx)
            latent = self.kv_adapt(self.bus.require("latent", self.layer_idx))
            rotary = self.bus.require("rotary", self.layer_idx)
            # New routing over borrowed keys: this layer's own view of what matters.
            allowed, effective = self.route(hidden_states, index_keys, queries)
            self.bus.topk = allowed
        else:
            query, = self._project(hidden_states)
            latent = self.kv_adapt(self.bus.require("latent", self.layer_idx))
            rotary = self.bus.require("rotary", self.layer_idx)
            allowed = self.bus.require("topk", self.layer_idx)
            index_keys, effective = None, None

        self.last_allowed = allowed.detach()
        return self._attend(hidden_states, latent, rotary, self.block_mask(allowed),
                            effective, index_keys, position_embeddings, query)

    def router_columns(self, effective, index_keys):
        """Extra query and key columns carrying the router's score into the logits.

        The indexer's gradient has to come from somewhere, and a mask cannot supply it.
        Adding the score inside the kernel through ``score_mod`` works and costs five
        times the attention: measured 28.40 ms against 5.57 ms, at every tile shape, so
        it is the per-element indirect load rather than the smaller tile it forces.

        Appending the score's own factors to the query and key instead puts it in the
        existing GEMM for free. Both sides are L2-normalized and the extra columns are
        pre-divided by ``scaling``, so the contribution the kernel adds to each logit is
        exactly ``index_gate * cos(effective, index_key)`` -- bounded by ``index_gate``
        however large the raw projections grow, which is the property a raw dot product
        would not have.
        """
        normalize = torch.nn.functional.normalize
        query = normalize(effective.float(), dim=-1) * (self.index_gate / self.scaling)
        return (query.to(effective.dtype).unsqueeze(1),
                normalize(index_keys.float(), dim=-1).to(index_keys.dtype).unsqueeze(1))

    def _attend(self, hidden_states, latent, rotary, mask, effective, index_keys,
                position_embeddings, projected_query=None):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if projected_query is None:
            projected_query = self.q_proj(hidden_states)
        query_states, gate = torch.chunk(
            projected_query.view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)

        projected = self.kv_b_proj(latent).view(
            *input_shape, self.num_heads, self.content_dim + self.head_dim)
        content_key, value_states = torch.split(
            projected, [self.content_dim, self.head_dim], dim=-1)
        content_key = self.k_norm(content_key)

        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        cos, sin = position_embeddings
        # Only the shared slice is position-dependent, so it is the only thing rotated.
        query_states, rotary = apply_rotary_pos_emb(
            query_states, rotary.unsqueeze(1), cos, sin)
        shared = rotary.expand(input_shape[0], self.num_heads, input_shape[1],
                               self.rope_dim)
        key_states = torch.cat([shared, content_key.transpose(1, 2)], dim=-1)
        value_states = value_states.transpose(1, 2)

        # FlexAttention compiles the block mask into a fused kernel that skips whole
        # blocks. DeepSeek's own sparse kernels are SM90/SM100 and this card is SM86, so
        # borrowing them is not available; this reaches the same place through Triton.
        # Measured against the alternatives at this shape: 5.51 ms here, 10.33 ms for
        # dense SDPA, 17.39 ms for SDPA with a token-level boolean mask.
        if effective is not None:
            heads, length = self.num_heads, input_shape[1]
            query_extra, key_extra = self.router_columns(effective, index_keys)
            query_states = torch.cat(
                [query_states, query_extra.expand(-1, heads, length, -1)], dim=-1)
            key_states = torch.cat(
                [key_states, key_extra.expand(-1, heads, length, -1)], dim=-1)

        attn_output = _flex()(query_states, key_states, value_states,
                              block_mask=mask, scale=self.scaling)

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output), None

    def routing_statistics(self):
        """What the last forward's routing looked like, or ``None`` if none has run.

        Three numbers, because raw density cannot tell a working router from a dead one:
        the local window and the diagonal are open regardless of score, so a router that
        has collapsed onto them still reports a healthy-looking density.

        * ``density`` -- allowed blocks over causally reachable blocks, the sparsity the
          kernel actually sees.
        * ``selected`` -- of the blocks top-k was free to choose, the share it chose. Its
          floor is ``keep / eligible``, and it moves only if different query blocks pick
          different keys.
        * ``entropy`` -- Shannon entropy of which key blocks got chosen, over log of the
          number of eligible keys, so 1.0 is spread evenly and 0.0 is every query block
          choosing the same key block. This is the collapse detector.
        """
        allowed = self.last_allowed
        if allowed is None:
            return None
        blocks = allowed.shape[-1]
        rows = torch.arange(blocks, device=allowed.device)
        offsets = rows.view(-1, 1) - rows.view(1, -1)
        reachable = (offsets >= 0).unsqueeze(0)
        eligible = (offsets > self.local_blocks).unsqueeze(0)

        chosen = (allowed & eligible).float()
        histogram = chosen.sum(dim=(0, 1))
        total = histogram.sum()
        # Normalized against the key blocks top-k *could* have reached, not against the
        # ones it did: dividing by the latter scores an even split over two blocks as a
        # perfect 1.0, which is the collapse this is meant to catch.
        available = max(2, blocks - self.local_blocks - 1)
        entropy = 0.0
        if total > 0:
            share = histogram[histogram > 0] / total
            entropy = float(-(share * share.log()).sum() / math.log(available))
        eligible_total = float(eligible.expand_as(allowed).sum())
        return {
            "mode": self.mode,
            "blocks": blocks,
            "density": float(allowed.sum()) / float(reachable.expand_as(allowed).sum()),
            "selected": (float(chosen.sum()) / eligible_total) if eligible_total else 0.0,
            "entropy": entropy,
        }

    def cached_numbers_per_token(self) -> int:
        """What this layer *would* add to a compressed serving cache.

        Aspirational for this class: CSA2 refuses `past_key_values` outright, so nothing
        here is cached at all today. The number is what the latent form costs, for
        comparison against GQA's 256, not a measurement of a cache that exists.
        """
        if self.mode != "full":
            return 0
        return self.latent + self.rope_dim + self.index_dim
