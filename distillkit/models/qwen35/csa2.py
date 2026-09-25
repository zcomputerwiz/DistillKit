"""Compressed Sparse Attention 2 over an MLA latent: DeepSeek-V4.1's CSA2 at ratio 1.

DeepSeek-V4.1-Flash (arXiv 2609.19969, section 2.3) defines CSA2: a lightweight indexer
scores the main KV entries and each query attends to its top-k of them, and every layer
takes one of three static modes that share state across depth:

* **Full** -- its own main KV and indexer keys, its own top-k
* **Reindex** -- borrows the main KV and indexer keys of the nearest Full layer, and scores
  them with its own indexer queries for a fresh top-k
* **Reuse** -- borrows the main KV and the nearest routing layer's top-k, and computes
  neither

Only the queries and the up-projections are ever private. CSA2 compresses every `m`
tokens into one entry and names `m = 1` as a supported case; this is that case, with the
main KV being MLA's latent and decoupled rotary key rather than V4.1's own entry. The
indexer keys are projected off that latent, as V4.1's are off its main KV, and their
position half is the rotary key the cache already holds, so they cost no cache.

**Selection, which llama.cpp has to reproduce exactly.** Per query token: the index score
of every causal position is `sum_h softplus(indexer_proj(x))_h * relu(q_h . k)`, the
top-`top_k` positions with a score above zero are read, and so are the last
`local_window` positions whatever their score, each position once. Three details differ
from the V4.1 graph and are deliberate:

* the head weights pass a softplus, where the reference scales a raw projection;
* a position scored exactly 0 is never picked, because zeros tie and a tie broken by
  topk's order differs between torch and `ggml_top_k`;
* the local window reads the shared latent rather than a separate per-layer SWA cache,
  so a selected position inside it is read once, not twice.

**Paths.** `_forward_tokens` takes every whole-sequence forward and keeps memory linear in
the sequence: the selection is held as positions and each query chunk's attention is
recomputed in backward. `_forward_gathered` materializes `[query, key]` and takes a decode
step, dense routing and the router bias.

**Training the indexer.** DeepSeek's recipe: a warm-up that distills the dense attention
into the indexer (`recorded_attention` under `dense_routing`), then a sparse stage in which
the indexer fits the attention over the positions it selected and is cut from the
backbone both ways (`isolated_indexer`). The router bias -- the index score folded into
the attention logits -- predates that recipe and is off in every current checkpoint.
"""

from __future__ import annotations

import contextlib
import math

import torch
from torch import nn

from .mla import Qwen35LatentAttention

__all__ = ["SparseIndexBus", "Qwen35SparseLatentAttention", "csa2_modes", "dense_routing",
           "isolated_indexer", "recorded_attention", "router_parameters",
           "routing_report"]

MODES = ("full", "reindex", "reuse")

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


#: Every parameter the router owns. One list, because the same filter written twice
#: drifts: `".index_"` reads as "the indexer's" and silently misses `indexer_proj`, which
#: is how `--freeze-router` came to leave the token-dependent head weights training.
ROUTER_PARAMETER_PARTS = ("index_q_proj", "index_k_proj", "index_weight",
                          "indexer_proj", "index_gate")


def router_parameters(module):
    """``(name, parameter)`` for every router parameter under ``module``."""
    return [(name, parameter) for name, parameter in module.named_parameters()
            if any(part in name for part in ROUTER_PARAMETER_PARTS)]


@contextlib.contextmanager
def dense_routing(model):
    """Run ``model`` with every causal block open and the router bias off its logits.

    A converted model's own attention is the teacher a self-distillation wants -- the
    source's attention is a different model's, fitted away by the key/value refit -- but
    reading it needs the routing out of the way, or the teacher is the student's router
    looking at itself.
    """
    layers = [m for m in model.modules() if isinstance(m, Qwen35SparseLatentAttention)]
    for module in layers:
        module.dense_routing = True
    try:
        yield model
    finally:
        for module in layers:
            module.dense_routing = False


@contextlib.contextmanager
def recorded_attention(model):
    """Keep each routing layer's attention distribution, which is the indexer's target.

    DeepSeek trains the indexer to imitate the attention it sits in front of: "to align
    the indexer outputs with the main attention distribution", by a KL against the
    per-query distribution summed over heads and normalized to one. That target is the
    model's own attention, so it costs a forward and no teacher.

    Recording it is not free -- the distribution is `[query, key]` per layer, which is the
    thing sparse attention exists to avoid materializing -- so it is a context rather than
    a flag left on. Open it with `dense_routing` to get the distribution the indexer is
    supposed to predict rather than the one its own selection already shaped.
    """
    layers = [m for m in model.modules() if isinstance(m, Qwen35SparseLatentAttention)]
    for module in layers:
        module.record_attention = True
    try:
        yield model
    finally:
        for module in layers:
            module.record_attention = False
            module.last_attention = None


@contextlib.contextmanager
def isolated_indexer(model):
    """Cut the gradient between the indexer and the model, both ways.

    DeepSeek's sparse stage: "we detach the indexer input from the computational graph for
    separate optimization. The training signal of the indexer is from only L_I, while the
    optimization of the main model is according to only the language modeling loss."

    That is two cuts here rather than one, because this fork has a term the reference does
    not. Their selection is a mask and nothing else, so a language-modeling loss cannot
    reach the indexer at all and only the input needs detaching. This fork also folds the
    index score into the attention logits through extra query and key columns, which
    exists precisely so the loss *can* reach an indexer that top-k would otherwise leave
    gradient-free. Under this policy the indexer has its own objective and no longer needs
    that path, so the columns stay in the function and carry no gradient.

    Leaving either cut out is not a smaller version of the policy. Keeping the first means
    the indexer's KL perturbs the backbone; keeping the second means the language model
    trains the indexer against the target its KL is pulling it toward.
    """
    layers = [m for m in model.modules() if isinstance(m, Qwen35SparseLatentAttention)]
    for module in layers:
        module.isolated_indexer = True
    try:
        yield model
    finally:
        for module in layers:
            module.isolated_indexer = False


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
    """What a Full layer publishes and the borrowing modes read, keyed by its publisher.

    Held by the text model -- not by the config, which is serialized and shared between
    models -- and cleared at the start of every forward, so nothing survives between
    batches. A borrowing layer that finds nothing under its donor is a configuration
    error, not something to paper over with a fallback.

    This used to be one set of slots that each publisher overwrote, so a reader took
    whatever had been written most recently and the answer depended on execution order.
    That made gradient checkpointing unusable -- it re-runs a layer's forward during the
    backward pass, out of order, and a borrower would have read the wrong publisher and
    produced quietly wrong gradients -- and the model refused the combination rather than
    risk it. Since checkpointing is what a long sequence needs, and long sequences are
    the point of the architecture, the ordering assumption had to go rather than the
    feature.

    Every reader now names its publisher, which it knows statically from the mode
    sequence: `latent_donor` is the nearest Full layer at or before it, and `topk_donor`
    the nearest layer that routes at all, Full or Reindex -- two relationships, because a
    Reuse layer behind a Reindex one takes its latent from the Full layer further back
    and its selection from the Reindex layer in front of it. A re-run then writes its own
    key and reads its donor's, and replay order stops meaning anything.
    """

    __slots__ = ("latents", "selections")

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        # layer index -> (index_keys, latent, rotary), written by Full layers.
        self.latents = {}
        # layer index -> the block or token selection, written by anything that routes.
        self.selections = {}

    def publish(self, layer_idx: int, index_keys, latent, rotary) -> None:
        self.latents[layer_idx] = (index_keys, latent, rotary)

    def select(self, layer_idx: int, allowed) -> None:
        self.selections[layer_idx] = allowed

    def require_latent(self, donor: int, layer_idx: int):
        if donor not in self.latents:
            raise RuntimeError(
                "layer %d reads the latent published by layer %s, which has not run in "
                "this forward pass" % (layer_idx, donor))
        return self.latents[donor]

    def require_selection(self, donor: int, layer_idx: int):
        if donor not in self.selections:
            raise RuntimeError(
                "layer %d reads the selection made by layer %s, which has not run in "
                "this forward pass" % (layer_idx, donor))
        return self.selections[donor]


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
        # Who this layer reads from, resolved here rather than left to whoever wrote the
        # bus last. A Full layer is its own donor for both. A Reindex layer borrows the
        # latent and routes for itself. A Reuse layer borrows both -- but not necessarily
        # from the same place, because a Reindex layer between it and its Full donor
        # publishes a selection without publishing a latent.
        order = [i for i, kind in enumerate(config.layer_types)
                 if "linear" not in str(kind)]
        if layer_idx in order:
            modes = csa2_modes(config, len(order))
            position = order.index(layer_idx)
            self.latent_donor = next(
                (order[q] for q in range(position, -1, -1) if modes[q] == "full"), None)
            self.topk_donor = next(
                (order[q] for q in range(position, -1, -1) if modes[q] != "reuse"), None)
            if self.latent_donor is None or self.topk_donor is None:
                raise ValueError(
                    "csa2 layer %d is %s with no Full layer in front of it; a mode "
                    "sequence has to open on full" % (layer_idx, mode))
        else:
            # Built outside the stack it belongs to, which is what a unit test does to
            # exercise `route` on its own. There is no sequence to resolve against, so
            # the layer answers for itself and never reads anyone.
            self.latent_donor = self.topk_donor = layer_idx
        # The last forward's block selection, kept so a run can report whether its router
        # is still choosing anything. Bool at [batch, blocks, blocks] -- a few kilobytes,
        # detached, out of the graph.
        self._last_allowed = None
        # The token path's selection, `(positions, valid)`, which `last_allowed` expands
        # on demand: at long context the expanded form is the `[query, key]` tensor the
        # path exists to avoid.
        self.last_selection = None
        # Queries per chunk on the token path. Memory there is about `chunk * seq` per
        # layer rather than `seq^2`; at 1024 or fewer tokens it is one chunk.
        self.query_chunk = int(getattr(config, "csa2_query_chunk", 1024))
        # Whether the last forward selected positions or blocks. The two paths report the
        # same three numbers over different units, and reading one as the other would make
        # a decode step look like a collapsed router.
        self.last_token_routed = False
        self.dense_routing = False
        # DeepSeek's sparse stage trains the indexer from its own KL alone and the model
        # from the language-modeling loss alone. See `isolated_indexer`.
        self.isolated_indexer = False
        # Set inside `recorded_attention`, which is how the indexer gets its target.
        self.record_attention = False
        self.last_attention = None
        self.index_dim = int(getattr(config, "csa2_index_dim", 64))
        self.index_heads = int(getattr(config, "csa2_index_heads", 4))
        self.top_k = int(getattr(config, "csa2_top_k", 128))
        self.local_window = int(getattr(config, "csa2_local_window", 32))
        if self.local_window < 0:
            raise ValueError("csa2_local_window must not be negative; got %d"
                             % self.local_window)
        self.rope_index = bool(getattr(config, "csa2_rope_index", True))
        # Whether the index score is added to the attention logits as well as deciding the
        # selection. It exists because a discrete top-k carries no gradient and the
        # indexer would otherwise never learn; the reference instead gives the indexer its
        # own KL against the attention distribution, and adds nothing to the logits.
        #
        # Keeping both is worse than either. Measured on the 2B: the warm-up drives the
        # indexer's cross entropy from 162.3 to 32.3 and the model's held-out loss *up*,
        # 1.4948 to 1.5117, because every improvement in what the router predicts is also
        # a change to the logits it is predicting about. The default stays on for
        # checkpoints trained with it; a model that means to follow the reference's recipe
        # turns it off and trains the indexer by `indexer_kl.py` instead.
        self.router_bias = bool(getattr(config, "csa2_router_bias", True))
        # V4.1's hierarchical indexer was implemented on the block path and went with it.
        # A checkpoint configured for it would otherwise load and silently run without it.
        if int(getattr(config, "csa2_candidate_k", 0) or 0) > 0:
            raise ValueError(
                "csa2_candidate_k is set, but the candidate hierarchy was removed with the "
                "block-sparse path; re-implement it per token before enabling it")

        bias = config.attention_bias
        # Index keys are published by Full layers and borrowed by everyone else. They are
        # read off the latent rather than off the hidden state, which is the reference's
        # V4.1 shape and costs less on three counts: a 384-wide input instead of a
        # 2048-wide one, nothing extra in the cache because the latent is already there,
        # and a score computed from the same summary the attention will actually read.
        # The 64 numbers a token used to spend on its own index key are what made CSA2's
        # cache larger than plain MLA's -- 512 against 448 -- for a mechanism that is
        # supposed to make it smaller.
        if mode == "full":
            self.index_k_proj = nn.Linear(self.latent_dim, self.index_dim, bias=bias)
        # Reuse computes no routing at all, so it needs no index queries either.
        if mode in ("full", "reindex"):
            self.index_q_proj = nn.Linear(
                config.hidden_size, self.index_heads * self.index_width, bias=bias)
            # DeepSeek weights the per-head scores with a projection of the token --
            # `indexer_proj` in llama.cpp's V4/V4.1 graph, one weight per head per token.
            # This fork used a single learned vector shared by every token, which cannot
            # say "this head matters here"; `indexer_head_weights` keeps that form for
            # checkpoints trained with it.
            self.token_head_weights = bool(
                getattr(config, "csa2_token_head_weights", True))
            if self.token_head_weights:
                self.indexer_proj = nn.Linear(config.hidden_size, self.index_heads,
                                              bias=bias)
            else:
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
            # `post_init` walks the tree after every module is built and re-initializes
            # anything it is not told to leave alone, so without this the identity above
            # is overwritten by a normal draw at `initializer_range` -- diagonal mean
            # 0.000 where it should be 1.000. The adapter exists to start as "read the
            # donor's latent unchanged" and learn to differ; starting it random instead
            # makes the up-projection fitted against it undo a random matrix, which costs
            # conditioning the fit has no reason to spend.
            self.kv_adapt.weight._is_hf_initialized = True
            # A borrowing layer never projects its own latent, so the inherited down
            # projection is dead weight -- 73,728 parameters per layer that would ship in
            # every checkpoint and take no gradient.
            del self.kv_a_proj
            del self.kv_a_norm

    @property
    def last_allowed(self):
        if self._last_allowed is None and self.last_selection is not None:
            positions, valid = self.last_selection
            batch, seq, _ = positions.shape
            counts = torch.zeros(batch, seq, seq, dtype=torch.int16, device=positions.device)
            counts.scatter_add_(-1, positions, valid.to(torch.int16))
            self._last_allowed = counts > 0
        return self._last_allowed

    @last_allowed.setter
    def last_allowed(self, value):
        self._last_allowed = value
        self.last_selection = None

    @property
    def latent_dim(self) -> int:
        return self.latent

    @property
    def index_width(self) -> int:
        """How wide an index vector is once its position half is attached.

        The content half comes off the latent and carries no position; the other half is
        the decoupled rotary key MLA already caches, reused rather than duplicated. That
        is the whole reason nothing extra is stored: a historical index key would
        otherwise need its own rotary table to be rebuilt, which is exactly why it used
        to be roped on the way in and cached.
        """
        return self.index_dim + self.rope_dim

    def index_keys_from(self, latent, rotary):
        """The index keys for a whole history, rebuilt from what the cache already holds.

        `latent` is the compressed key/value summary and `rotary` the shared decoupled
        key, both roped and cached by the owning layer. Neither is widened and nothing
        else is read, so a Full layer's cache is the latent and the rotary slice and no
        more -- 448 numbers a token where it used to be 512, which is what stopped CSA2
        costing more to cache than the plain MLA it is built on.
        """
        return torch.cat([self.index_k_proj(latent), rotary], dim=-1)

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
            modules.append(self.kv_a_proj)
        if self.mode in ("full", "reindex"):
            modules.append(self.index_q_proj)
        if len(modules) == 1:
            return (self.q_proj(hidden_states),)
        weight = torch.cat([module.weight for module in modules], dim=0)
        bias = (torch.cat([module.bias for module in modules], dim=0)
                if modules[0].bias is not None else None)
        fused = torch.nn.functional.linear(hidden_states, weight, bias)
        return fused.split([module.out_features for module in modules], dim=-1)

    def _rope_index(self, tensor, position_embeddings):
        """Rotate the trailing slice of an index vector, the way the reference does.

        DeepSeek ropes both the indexer query and the indexer key, so a block's score
        carries how far away it is and not only what is in it. Without this the router
        sees content alone and has to infer distance from the local window, which is the
        one thing it is never asked to decide.

        The rotated slice is the *trailing* one here, matching the reference's nope-then-pe
        layout, and it is fed to the stock rotary on its own -- that function rotates the
        leading dimensions of whatever it is handed, so handing it the slice is enough.
        """
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

        cos, sin = position_embeddings
        # The index head can be narrower than the model's rotary width, so the tables are
        # cut to the slice as well. Cutting only the slice leaves the two disagreeing and
        # the rotary raises.
        width = min(cos.shape[-1], tensor.shape[-1])
        cos, sin = cos[..., :width], sin[..., :width]
        head = tensor if tensor.ndim == 4 else tensor.unsqueeze(1)
        nope, rotated = head[..., :-width], head[..., -width:]
        rotated, _ = apply_rotary_pos_emb(rotated, rotated, cos, sin)
        out = torch.cat([nope, rotated], dim=-1)
        return out if tensor.ndim == 4 else out.squeeze(1)

    def _rope_shared(self, rotary, position_embeddings):
        """Rotate the shared rotary key on its own, so the cache can hold it roped.

        MLA already stores its rotary slice this way: the slice goes in carrying its own
        position, and replaying it later needs no position bookkeeping. The gathered path
        relies on that, because at a decode step the only positions it is handed are the
        new token's.
        """
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

        if position_embeddings is None:
            return rotary
        cos, sin = position_embeddings
        head = rotary.unsqueeze(1)
        _, rotated = apply_rotary_pos_emb(head, head, cos, sin)
        return rotated.squeeze(1)

    def _select_tokens(self, index_queries, index_keys, weights, q_positions,
                       kv_positions):
        """Which cached positions each query token may read: one decision per token.

        This is the reference's shape and the reason it decodes. `build_lid_top_k` scores
        `[n_kv, n_tokens]` and takes `ggml_top_k` over *positions*, per query token; the
        block level is a pre-filter on the key axis that pools with the token count
        untouched, and it is skipped outright once it cannot bite. Nothing groups the
        queries, so a step with one of them is an ordinary case.

        ``top_k`` counts tokens. This is the dense form, for a decode step and for the
        warm-up; `_select_positions` is the same selection over a whole sequence, chunked.
        """
        scores = torch.einsum("bqhd,bkd->bhqk", index_queries.float(), index_keys.float())
        gain = weights.permute(0, 2, 1).unsqueeze(-1).float()
        scores = (torch.relu(scores) * gain).sum(dim=1)

        causal = kv_positions.view(1, 1, -1) <= q_positions.view(1, -1, 1)
        scores = scores.masked_fill(~causal, float("-inf"))
        keep = max(1, min(self.top_k, index_keys.shape[1]))
        picked = scores.topk(keep, dim=-1)
        allowed = torch.zeros_like(scores, dtype=torch.bool)
        # Only positions some head scored above zero. The score is a ReLU sum under
        # positive weights, so "no evidence" is exactly 0 and ties; breaking that tie by
        # topk's order would make the selection depend on tensor shape and on the
        # backend, so training and llama.cpp would read different positions. Dropping
        # zeros is one extra mask on both sides and measured free: under 0.0002 nats on
        # WikiText at 384 and 1024 tokens, within noise.
        allowed.scatter_(-1, picked.indices, picked.values > 0)
        # The recent window is not spent out of the top-k budget: a query that cannot see
        # its own immediate context produces gradients about the router rather than about
        # the architecture.
        offsets = q_positions.view(1, -1, 1) - kv_positions.view(1, 1, -1)
        allowed |= (offsets >= 0) & (offsets < max(self.local_window, 1))
        return allowed & causal

    def _attend_gathered(self, hidden_states, latent, rotary, allowed, effective,
                         index_keys, position_embeddings, projected_query=None):
        """Masked attention over the whole cached history, mask decided per query token.

        Every term matches ``_attend``, the router's logit bias included. Leaving that out
        would make a token read through the cache disagree with the same token read in a
        whole-sequence forward, which is the one property this path exists to keep.

        The rotary slice and the index keys arrive already rotated, because they come from
        a cache that was written at their own positions. Only the query is rotated here.
        """
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

        batch, seq, _ = hidden_states.shape
        kv_len = latent.shape[1]
        if projected_query is None:
            projected_query = self.q_proj(hidden_states)
        query_states, gate = torch.chunk(
            projected_query.view(batch, seq, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(batch, seq, -1)
        query_states = self.q_norm(
            query_states.view(batch, seq, -1, self.head_dim)).transpose(1, 2)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)

        # Read the latent where it stands rather than expanding it. The up-projection is
        # linear, so `q . (W_k c) = (W_k^T q) . c` and `sum_s a_s (W_v c_s) = W_v sum_s
        # a_s c_s`: the query moves into latent space once and the aggregate expands once,
        # and the cached history is never widened at all. A content key norm would put a
        # per-token scale between the two and there would be no such identity, which is
        # the other reason that norm is gone.
        #
        # It is a decode trade, not a free one. Expanding costs `kv * latent * heads *
        # (content + head_dim)` once; absorbing costs `heads * seq * content * latent` and
        # then carries `latent` rather than `head_dim` through the score. Past roughly 670
        # query tokens at this geometry the expansion is cheaper, and a prefill reads
        # exactly as many keys as it has queries -- so `seq < kv_len` is both the test and
        # the meaning: some of this history was written by an earlier call.
        if self.k_norm is None and seq < kv_len:
            return self._attend_absorbed(latent, rotary, query_states, gate, allowed,
                                         effective, index_keys, batch, seq, kv_len)

        projected = self.kv_b_proj(latent).view(
            batch, kv_len, self.num_heads, self.content_dim + self.head_dim)
        content_key, value_states = torch.split(
            projected, [self.content_dim, self.head_dim], dim=-1)
        if self.k_norm is not None:
            content_key = self.k_norm(content_key)
        shared = rotary.unsqueeze(1).expand(batch, self.num_heads, kv_len, self.rope_dim)
        key_states = torch.cat([shared, content_key.transpose(1, 2)], dim=-1)
        value_states = value_states.transpose(1, 2)

        if effective is not None:
            query_extra, key_extra = self.router_columns(effective, index_keys)
            query_states = torch.cat(
                [query_states, query_extra.expand(-1, self.num_heads, seq, -1)], dim=-1)
            key_states = torch.cat(
                [key_states, key_extra.expand(-1, self.num_heads, kv_len, -1)], dim=-1)

        mask = torch.zeros(allowed.shape, dtype=query_states.dtype,
                           device=query_states.device)
        mask = mask.masked_fill(~allowed, float("-inf")).unsqueeze(1)
        if self.record_attention:
            self._record(query_states, key_states, allowed)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=mask, scale=self.scaling)
        attn_output = attn_output.transpose(1, 2).reshape(batch, seq, -1).contiguous()
        return self.o_proj(attn_output * torch.sigmoid(gate)), None

    def _record(self, query_states, key_states, allowed):
        """The attention distribution the indexer is asked to predict.

        DeepSeek's target: the per-query distribution over preceding tokens, summed over
        heads and normalized to one. Summed rather than averaged and then normalized,
        because what the indexer has to rank is where the attention went in total, not how
        any single head split it.

        Under `no_grad` and detached. It is a target, and the backward it would otherwise
        carry would run through the very attention whose selection is being trained.
        """
        with torch.no_grad():
            scores = torch.einsum("bhqd,bhkd->bhqk", query_states.float(),
                                  key_states.float()) * self.scaling
            scores = scores.masked_fill(~allowed.unsqueeze(1), float("-inf"))
            pooled = scores.softmax(-1).sum(1)
            self.last_attention = (
                pooled / pooled.sum(-1, keepdim=True).clamp_min(1e-9)).detach()

    def token_scores(self, hidden_states, index_keys, queries=None,
                     position_embeddings=None, cache_position=None):
        """The indexer's own scores over positions, with gradient, for its KL.

        `route` reads these under `no_grad` because a selection carries none. Training the
        indexer reads them directly: the score *is* the thing being fitted, against an
        attention distribution over the same positions.
        """
        batch, seq, _ = hidden_states.shape
        kv_len = index_keys.shape[1]
        index_queries, weights = self._index_queries(
            hidden_states, queries, position_embeddings)
        scores = torch.einsum("bqhd,bkd->bhqk", index_queries.float(),
                              index_keys.float())
        gain = weights.permute(0, 2, 1).unsqueeze(-1).float()
        # Scaled the way attention scales, and for the same reason: a raw dot product over
        # this width reaches 123 with a standard deviation of 12, and a softmax over that
        # is a one-hot on an arbitrary position. The KL then measures the scale rather
        # than the ranking -- it settled at 21 to 32 nats where a prediction that knows
        # nothing scores 5.94. Selection is invariant to a positive scale, so this changes
        # what the objective sees and nothing about what the router does.
        scores = (torch.relu(scores) * gain).sum(dim=1) * self.index_width ** -0.5

        if cache_position is None:
            cache_position = torch.arange(kv_len - seq, kv_len,
                                          device=hidden_states.device)
        positions = torch.arange(kv_len, device=hidden_states.device)
        causal = positions.view(1, 1, -1) <= cache_position.view(1, -1, 1)
        return scores, causal

    def _attend_absorbed(self, latent, rotary, query_states, gate, allowed, effective,
                         index_keys, batch, seq, kv_len):
        """Attention read against the cached latent, with the up-projection folded in.

        The same function as the expanding path, by an identity rather than by
        approximation, so the two agree to arithmetic noise. The score splits the way the
        key does -- a rotary slice shared by every head and a content half that reaches
        the latent through `W_k` -- and the aggregate comes back out through `W_v`.

        This is what makes the measured cache saving a serving saving. Without it the
        up-projection runs over the whole cached history at every step: at 32K that is a
        384 to 3584 matrix multiply over 32,768 tokens per layer per token generated, and
        it costs more than the memory the compression saved.
        """
        weight = self.kv_b_proj.weight.view(
            self.num_heads, self.content_dim + self.head_dim, self.latent)
        w_key, w_value = weight[:, :self.content_dim], weight[:, self.content_dim:]

        query_rope = query_states[..., :self.rope_dim]
        query_content = query_states[..., self.rope_dim:]
        absorbed = torch.einsum("bhqc,hcl->bhql", query_content.float(), w_key.float())
        scores = (torch.einsum("bhql,bkl->bhqk", absorbed, latent.float())
                  + torch.einsum("bhqr,bkr->bhqk", query_rope.float(), rotary.float()))
        if effective is not None:
            # The router's columns are a rank-`index_dim` term in the same logits, and
            # they carry their own 1/scaling, so they join before the scale like the rest.
            query_extra, key_extra = self.router_columns(effective, index_keys)
            scores = scores + torch.einsum(
                "bxqi,bxki->bqk", query_extra.float(), key_extra.float()).unsqueeze(1)
        scores = scores * self.scaling
        scores = scores.masked_fill(~allowed.unsqueeze(1), float("-inf"))

        probabilities = scores.softmax(-1).to(latent.dtype)
        aggregate = torch.einsum("bhqk,bkl->bhql", probabilities, latent)
        attn_output = torch.einsum("bhql,hvl->bhqv", aggregate, w_value)
        attn_output = attn_output.transpose(1, 2).reshape(batch, seq, -1).contiguous()
        return self.o_proj(attn_output * torch.sigmoid(gate)), None

    def _index_queries(self, hidden_states, queries, position_embeddings):
        """Roped index queries and their per-head weights, for the gathered path."""
        batch, seq, _ = hidden_states.shape
        if self.isolated_indexer:
            hidden_states = hidden_states.detach()
            queries = None if queries is None else queries.detach()
        if queries is None:
            # The fused projection hands these down in a forward. A caller that wants the
            # scores on their own -- the warm-up's KL, which never runs the attention --
            # has only the hidden state, and projecting here is what lets `index_q_proj`
            # take a gradient from that objective.
            queries = self.index_q_proj(hidden_states)
        index_queries = queries.view(batch, seq, self.index_heads, self.index_width)
        if self.rope_index and position_embeddings is not None:
            # Only the query is rotated now. `_rope_index` turns the trailing slice, which
            # is exactly the half that meets the cached rotary key; the leading half meets
            # a content key that carries no position and must not be turned.
            index_queries = self._rope_index(
                index_queries.transpose(1, 2), position_embeddings).transpose(1, 2)
        if self.token_head_weights:
            weights = torch.nn.functional.softplus(self.indexer_proj(hidden_states))
        else:
            weights = torch.nn.functional.softplus(self.index_weight)
            weights = weights.view(1, 1, -1).expand(batch, seq, -1)
        return index_queries, weights

    def _forward_gathered(self, hidden_states, position_embeddings, past_key_values,
                          cache_position):
        """Dense per-token routing: a decode step, dense routing, and the router bias.

        Materializes `[query, key]`, which is fine for a decode step's one row and for the
        short warm-up, and is what `_forward_tokens` avoids for everything else.
        """
        batch, seq, _ = hidden_states.shape
        if self.mode == "full":
            query, compressed, queries = self._project(hidden_states)
            latent, rotary = torch.split(compressed, [self.latent, self.rope_dim], dim=-1)
            latent = self.kv_a_norm(latent)
            rotary = self._rope_shared(rotary, position_embeddings)
            if past_key_values is not None:
                # The latent and the rotary slice, and nothing else. The index keys used
                # to ride in this slot because they had been roped on the way in and could
                # not be rebuilt without their own rotary table; taking their position half
                # from the rotary key that is already here removes the need to store them.
                latent, rotary = past_key_values.update(
                    latent.unsqueeze(1), rotary.unsqueeze(1), self.layer_idx)
                latent, rotary = latent.squeeze(1), rotary.squeeze(1)
            index_keys = self.index_keys_from(latent, rotary)
            self.bus.publish(self.layer_idx, index_keys, latent, rotary)
        elif self.mode == "reindex":
            query, queries = self._project(hidden_states)
            index_keys, borrowed, rotary = self.bus.require_latent(
                self.latent_donor, self.layer_idx)
            latent = self.kv_adapt(borrowed)
        else:
            query, = self._project(hidden_states)
            _, borrowed, rotary = self.bus.require_latent(
                self.latent_donor, self.layer_idx)
            latent = self.kv_adapt(borrowed)
            index_keys, queries = None, None

        kv_len = latent.shape[1]
        if cache_position is None:
            cache_position = torch.arange(kv_len - seq, kv_len, device=latent.device)
        kv_positions = torch.arange(kv_len, device=latent.device)

        if self.mode == "reuse":
            allowed = self.bus.require_selection(self.topk_donor, self.layer_idx)
            effective = None
        else:
            index_queries, weights = self._index_queries(
                hidden_states, queries, position_embeddings)
            if self.dense_routing:
                allowed = (kv_positions.view(1, 1, -1)
                           <= cache_position.view(1, -1, 1)).expand(batch, seq, kv_len)
                effective = None
            else:
                with torch.no_grad():
                    allowed = self._select_tokens(
                        index_queries, index_keys, weights, cache_position, kv_positions)
                effective = ((index_queries * weights.unsqueeze(-1)).sum(dim=2)
                             if self.router_bias else None)
            self.bus.select(self.layer_idx, allowed)

        self.last_allowed = allowed.detach()
        self.last_token_routed = True
        return self._attend_gathered(hidden_states, latent, rotary, allowed, effective,
                                     index_keys, position_embeddings, query)

    def _token_path(self, past_key_values) -> bool:
        """Whether this forward takes `_forward_tokens`, the one no-cache path.

        Per-token selection is what the llama.cpp graph (`build_lid_top_k`) computes, so
        a whole-sequence forward -- evaluation, prefill, training -- takes it. What stays
        on the dense gathered path: a decode step (it has a cache), dense routing (the
        warm-up's teacher), and the router's logit bias, which this path does not carry.

        There used to be a third path, block-sparse FlexAttention, taken whenever a length
        divided 128. It routed 128-token blocks, which neither training nor serving does,
        and scored the scale checkpoint 0.025 nats worse on WikiText; it was removed.
        """
        return (past_key_values is None and not self.dense_routing
                and not self.router_bias)

    def _forward_tokens(self, hidden_states, position_embeddings):
        """Per-token routing over a whole sequence, a chunk of queries at a time.

        The same function as `_forward_gathered` with no cache: each query reads the
        top-`top_k` positions by index score plus its last `local_window`. Nothing
        `[query, key]`-sized outlives a chunk -- the selection is kept as positions, the
        attention of each chunk is recomputed in backward rather than stored, and the
        indexer's target is kept only over the positions it was selected on -- so memory
        grows with the sequence rather than its square. At 1024 tokens and the default
        chunk that is one chunk, the computation `_forward_gathered` does.

        ponytail: each chunk's attention is dense SDPA under a mask, so compute is still
        O(seq^2); a gather over the selected positions would make it O(seq * top_k).
        """
        batch, seq, _ = hidden_states.shape
        if self.mode == "full":
            query, compressed, queries = self._project(hidden_states)
            latent, rotary = torch.split(compressed, [self.latent, self.rope_dim], dim=-1)
            latent = self.kv_a_norm(latent)
            rotary = self._rope_shared(rotary, position_embeddings)
            index_keys = self.index_keys_from(latent, rotary)
            self.bus.publish(self.layer_idx, index_keys, latent, rotary)
        elif self.mode == "reindex":
            query, queries = self._project(hidden_states)
            index_keys, borrowed, rotary = self.bus.require_latent(
                self.latent_donor, self.layer_idx)
            latent = self.kv_adapt(borrowed)
        else:
            query, = self._project(hidden_states)
            _, borrowed, rotary = self.bus.require_latent(self.latent_donor, self.layer_idx)
            latent = self.kv_adapt(borrowed)

        if self.mode == "reuse":
            positions, valid = self.bus.require_selection(self.topk_donor, self.layer_idx)
        else:
            index_queries, weights = self._index_queries(
                hidden_states, queries, position_embeddings)
            positions, valid = self._select_positions(index_queries, index_keys, weights)
            self.bus.select(self.layer_idx, (positions, valid))
        self.last_allowed = None
        self.last_selection = (positions, valid)
        self.last_token_routed = True

        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        query_states, gate = torch.chunk(
            query.view(batch, seq, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(batch, seq, -1)
        query_states = self.q_norm(
            query_states.view(batch, seq, -1, self.head_dim)).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)
        projected = self.kv_b_proj(latent).view(
            batch, seq, self.num_heads, self.content_dim + self.head_dim)
        content_key, value_states = torch.split(
            projected, [self.content_dim, self.head_dim], dim=-1)
        if self.k_norm is not None:
            content_key = self.k_norm(content_key)
        shared = rotary.unsqueeze(1).expand(batch, self.num_heads, seq, self.rope_dim)
        key_states = torch.cat([shared, content_key.transpose(1, 2)], dim=-1)
        value_states = value_states.transpose(1, 2)

        from torch.utils.checkpoint import checkpoint

        outputs, targets = [], []
        for start in range(0, seq, self.query_chunk):
            end = min(seq, start + self.query_chunk)
            q, k, v = query_states[:, :, start:end], key_states[:, :, :end], value_states[:, :, :end]
            where, ok = positions[:, start:end], valid[:, start:end]
            if self.record_attention:
                targets.append(self._record_chunk(q, k, where, ok, start))
            outputs.append(checkpoint(self._attend_chunk, q, k, v, where, ok, start,
                                      use_reentrant=False))
        if self.record_attention:
            self.last_attention = (positions, valid, torch.cat(targets, dim=1))
        attn_output = torch.cat(outputs, dim=2).transpose(1, 2).reshape(batch, seq, -1)
        return self.o_proj(attn_output * torch.sigmoid(gate)), None

    def _select_positions(self, index_queries, index_keys, weights):
        """`(positions, valid)`, `[batch, seq, top_k + local_window]`: what each query reads.

        The first `top_k` columns are the top-k over every causal position, exactly as
        `_select_tokens` takes them, and the rest are the local window, which is not spent
        out of the budget. A top-k pick that falls inside the window is marked invalid
        rather than kept twice, so `valid` marks the union once -- the set
        `_select_tokens` returns as a mask. Scored a chunk of queries at a time under
        no_grad, so the `[query, key]` scores never exist whole.
        """
        batch, seq = index_queries.shape[:2]
        device = index_queries.device
        keep = max(1, min(self.top_k, seq))
        window = max(self.local_window, 1)
        top = torch.zeros(batch, seq, keep, dtype=torch.long, device=device)
        chosen = torch.zeros(batch, seq, keep, dtype=torch.bool, device=device)
        gain = weights.permute(0, 2, 1).unsqueeze(-1).float()
        with torch.no_grad():
            for start in range(0, seq, self.query_chunk):
                end = min(seq, start + self.query_chunk)
                scores = torch.einsum("bqhd,bkd->bhqk", index_queries[:, start:end].float(),
                                      index_keys[:, :end].float())
                scores = (torch.relu(scores) * gain[:, :, start:end]).sum(dim=1)
                rows = torch.arange(start, end, device=device).view(-1, 1)
                causal = torch.arange(end, device=device).view(1, -1) <= rows
                scores = scores.masked_fill(~causal, float("-inf"))
                picked = scores.topk(min(keep, end), dim=-1)
                width = picked.indices.shape[-1]
                top[:, start:end, :width] = picked.indices
                # A pick at -inf is a row with fewer causal positions than the budget, and
                # one at exactly 0 is a position no head scored: those tie, and which of
                # them fills a budget is whatever topk does at this shape -- ggml_top_k
                # does something else. So neither is ever picked; see `_select_tokens`.
                # One inside the window is already read by the window's own columns.
                chosen[:, start:end, :width] = ((picked.values > 0)
                                                & (rows - picked.indices >= window))
            rows = torch.arange(seq, device=device).view(1, -1, 1)
            local = rows - torch.arange(window, device=device).view(1, 1, -1)
            positions = torch.cat([top, local.clamp_min(0).expand(batch, -1, -1)], dim=-1)
            valid = torch.cat([chosen, (local >= 0).expand(batch, -1, -1)], dim=-1)
        return positions, valid

    def _chunk_allowed(self, where, ok, end):
        """`[batch, chunk, end]` booleans for the queries `start:` from their positions."""
        batch, rows, _ = where.shape
        # Added rather than scattered: an invalid column points at 0 as well, and a
        # scatter of True and False to the same place keeps whichever lands last.
        counts = torch.zeros(batch, rows, end, dtype=torch.int16, device=where.device)
        counts.scatter_add_(-1, where, ok.to(torch.int16))
        return counts > 0

    def _attend_chunk(self, q, k, v, where, ok, start):
        allowed = self._chunk_allowed(where, ok, k.shape[2])
        mask = torch.zeros(allowed.shape, dtype=q.dtype, device=q.device)
        mask = mask.masked_fill(~allowed, float("-inf")).unsqueeze(1)
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=self.scaling)

    def _record_chunk(self, q, k, where, ok, start):
        """`_record`'s target for one chunk, kept only at the positions read.

        Summed over heads and normalized over the allowed set, as `_record` does; since
        the positions cover that set exactly once, the kept values still sum to one.
        """
        with torch.no_grad():
            allowed = self._chunk_allowed(where, ok, k.shape[2])
            scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * self.scaling
            scores = scores.masked_fill(~allowed.unsqueeze(1), float("-inf"))
            pooled = scores.softmax(-1).sum(1)
            pooled = pooled / pooled.sum(-1, keepdim=True).clamp_min(1e-9)
            return pooled.gather(-1, where).masked_fill(~ok, 0.0)

    def token_loss(self, hidden_states, index_keys, positions, valid, target,
                   position_embeddings=None, query_mask=None):
        """`(sum, count)` of the indexer's cross entropy against a compact target.

        The selected-set KL `indexer_loss` takes over `[query, key]` scores, computed
        only at the positions each query read, a chunk at a time with the chunk's scores
        recomputed in backward. `index_keys` carries the gradient to the donor's key
        projection, as the dense form's does.
        """
        from torch.utils.checkpoint import checkpoint

        queries, weights = self._index_queries(hidden_states, None, position_embeddings)
        batch, seq = queries.shape[:2]
        total, count = queries.new_zeros((), dtype=torch.float32), 0
        for start in range(0, seq, self.query_chunk):
            end = min(seq, start + self.query_chunk)
            rows = None if query_mask is None else query_mask[:, start:end].to(torch.bool)
            if rows is not None and not bool(rows.any()):
                continue
            per_query = checkpoint(self._token_loss_chunk, queries[:, start:end],
                                   weights[:, start:end], index_keys, positions[:, start:end],
                                   valid[:, start:end], target[:, start:end],
                                   use_reentrant=False)
            if rows is not None:
                per_query = per_query[rows]
            total = total + per_query.sum()
            count += per_query.numel()
        return total, count

    def _token_loss_chunk(self, queries, weights, index_keys, where, ok, target):
        batch, rows, width = where.shape
        flat = where + (torch.arange(batch, device=where.device) * index_keys.shape[1]).view(-1, 1, 1)
        keys = index_keys.reshape(-1, index_keys.shape[-1])[flat.view(-1)].view(
            batch, rows, width, -1)
        scores = torch.einsum("bqhd,bqsd->bhqs", queries.float(), keys.float())
        gain = weights.permute(0, 2, 1).unsqueeze(-1).float()
        scores = (torch.relu(scores) * gain).sum(dim=1) * self.index_width ** -0.5
        predicted = torch.log_softmax(scores.masked_fill(~ok, float("-inf")), dim=-1)
        return -(target * predicted.masked_fill(~ok, 0.0)).sum(-1)

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kwargs):
        batch, seq, _ = hidden_states.shape
        if self.bus is None:
            raise RuntimeError(
                "layer %d has no SparseIndexBus; the text model injects one after it "
                "builds its layers" % self.layer_idx)
        if self._token_path(past_key_values):
            return self._forward_tokens(hidden_states, position_embeddings)
        self.last_selection = None
        return self._forward_gathered(hidden_states, position_embeddings,
                                      past_key_values, kwargs.get("cache_position"))

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
        if self.isolated_indexer:
            # DeepSeek's second cut, at the one place both paths pass through. These
            # columns are a real term in the logits and they stay in the function; what
            # stops is the language-modeling loss reaching the indexer along them, which
            # would train it against the target its own KL is pulling it toward. The
            # reference needs no equivalent, because there the selection is a mask and
            # there is no path from the loss to the indexer to cut.
            effective, index_keys = effective.detach(), index_keys.detach()
        normalize = torch.nn.functional.normalize
        # `index_gate` is an indexer parameter and the KL never touches it, so under the
        # policy it takes no gradient from anywhere and holds where it was left. That is
        # the closest this fork gets to the reference, which has no such scalar because it
        # has no columns for one to scale.
        gate = self.index_gate.detach() if self.isolated_indexer else self.index_gate
        query = normalize(effective.float(), dim=-1) * (gate / self.scaling)
        return (query.to(effective.dtype).unsqueeze(1),
                normalize(index_keys.float(), dim=-1).to(index_keys.dtype).unsqueeze(1))

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
        if self.last_selection is not None:
            return self._compact_statistics()
        allowed = self.last_allowed
        if allowed is None:
            return None
        blocks = allowed.shape[-1]
        near = self.local_window
        columns = torch.arange(blocks, device=allowed.device)
        query_rows = torch.arange(blocks - allowed.shape[-2], blocks, device=allowed.device)
        offsets = query_rows.view(-1, 1) - columns.view(1, -1)
        reachable = (offsets >= 0).unsqueeze(0)
        eligible = (offsets > near).unsqueeze(0)

        chosen = (allowed & eligible).float()
        histogram = chosen.sum(dim=(0, 1))
        total = histogram.sum()
        # Normalized against the key blocks top-k *could* have reached, not against the
        # ones it did: dividing by the latter scores an even split over two blocks as a
        # perfect 1.0, which is the collapse this is meant to catch.
        available = max(2, blocks - near - 1)
        entropy = 0.0
        if total > 0:
            share = histogram[histogram > 0] / total
            entropy = float(-(share * share.log()).sum() / math.log(available))
        eligible_total = float(eligible.expand_as(allowed).sum())
        return {
            "mode": self.mode,
            "unit": "positions",
            "blocks": blocks,
            "density": float(allowed.sum()) / float(reachable.expand_as(allowed).sum()),
            "selected": (float(chosen.sum()) / eligible_total) if eligible_total else 0.0,
            "entropy": entropy,
        }

    def _compact_statistics(self):
        """`routing_statistics` read off the token path's positions, without expanding them.

        The same three numbers over the same sets: the window is `offset < local_window`,
        a top-k pick counts as chosen only beyond it (`offset > local_window`), and the
        positions mark the union once.
        """
        positions, valid = self.last_selection
        batch, seq, _ = positions.shape
        near = self.local_window
        rows = torch.arange(seq, device=positions.device).view(1, -1, 1)
        top = positions[..., :-max(near, 1)]
        picked = valid[..., :top.shape[-1]] & (rows - top > near)
        histogram = torch.bincount(top[picked], minlength=seq).float()
        total = histogram.sum()
        available = max(2, seq - near - 1)
        entropy = 0.0
        if total > 0:
            share = histogram[histogram > 0] / total
            entropy = float(-(share * share.log()).sum() / math.log(available))
        reachable = batch * seq * (seq + 1) / 2
        beyond = torch.arange(seq, device=positions.device) - near
        eligible_total = float(batch * beyond.clamp_min(0).sum())
        return {
            "mode": self.mode, "unit": "positions", "blocks": seq,
            "density": float(valid.sum()) / reachable,
            "selected": float(total) / eligible_total if eligible_total else 0.0,
            "entropy": entropy,
        }

    def cached_numbers_per_token(self) -> int:
        """What this layer adds to the compressed cache per token.

        A full layer writes the latent and the decoupled rotary key. The index keys are
        rebuilt from both rather than stored -- their content half is a projection of the
        latent and their position half *is* the rotary key -- so they cost nothing here.
        Storing them is what used to make this 512 against plain MLA's 448, so CSA2 cost
        more to cache than the thing it compresses. A borrowing layer writes nothing at
        all: it reads its donor's, which is where the second halving comes from.
        """
        if self.mode != "full":
            return 0
        return self.latent + self.rope_dim
