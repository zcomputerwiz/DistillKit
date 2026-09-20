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

import contextlib
import math

import torch
from torch import nn

from .mla import Qwen35LatentAttention

__all__ = ["SparseIndexBus", "Qwen35SparseLatentAttention", "csa2_modes", "dense_routing",
           "isolated_indexer", "router_parameters", "routing_report"]

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

    __slots__ = ("latents", "selections", "candidates", "candidate_index")

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        # layer index -> (index_keys, latent, rotary), written by Full layers.
        self.latents = {}
        # layer index -> the block or token selection, written by anything that routes.
        self.selections = {}
        # Level one of the hierarchy, published by at most one layer and read by the
        # layers after it. Stays None when the hierarchy is off, which is the default.
        # The indices are what a reader gathers; the mask is what a report reads.
        self.candidates = None
        self.candidate_index = None

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
        self.last_allowed = None
        self.last_candidates = None
        self.last_candidate_index = None
        # Whether the last forward selected positions or blocks. The two paths report the
        # same three numbers over different units, and reading one as the other would make
        # a decode step look like a collapsed router.
        self.last_token_routed = False
        self.dense_routing = False
        # DeepSeek's sparse stage trains the indexer from its own KL alone and the model
        # from the language-modeling loss alone. See `isolated_indexer`.
        self.isolated_indexer = False
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
        self.rope_index = bool(getattr(config, "csa2_rope_index", True))
        # Level one of the hierarchy. The named layer publishes a candidate block set and
        # the layers after it choose positions inside it. -1 leaves the hierarchy off, and
        # a layer only publishes if it routes at all -- a reuse layer has no scores.
        self.candidate_layer = int(getattr(config, "csa2_candidate_layer", -1))
        self.candidate_k = int(getattr(config, "csa2_candidate_k", 0))
        if self.candidate_k and self.candidate_k < self.top_k:
            raise ValueError(
                "csa2_candidate_k %d is narrower than csa2_top_k %d: level one would "
                "hand level two fewer blocks than it is asked to pick"
                % (self.candidate_k, self.top_k))

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

    def block_scores(self, hidden_states, index_keys, queries=None,
                     position_embeddings=None):
        """``(scores, eligible)``: what the router thinks of every block pair.

        ``route`` reads this under ``no_grad``, because top-k and a scatter carry no
        gradient and the selection is all it wants. Distillation reads it directly: when
        the target is a teacher's attention over the same blocks, the score itself is the
        thing being trained, and the logit fold that normally carries the indexer's
        gradient is not involved.

        Scored from the block's *leading* query against pooled key summaries, which is the
        same causal choice ``route`` documents -- one decision serves a whole block, so it
        may only use what the earliest token in that block can see.
        """
        blocks = hidden_states.shape[1] // self.block_size
        return self._score_blocks(*self._index_inputs(
            hidden_states, index_keys, queries, position_embeddings), blocks)

    def _score_blocks(self, queries, index_keys, weights, blocks, candidate_index=None):
        """Leading-query scores against pooled key summaries, and which pairs compete.

        One decision serves every token in the block, so it may only use what the
        *earliest* of them can see. Pooling the block's queries -- an amax over all of
        them, as this did -- lets the block's last token change what its first token is
        allowed to read: measured at 18 of 200 random single-token mutations, and 191 of
        200 at larger ones. Taking the leading query is the most informative choice that
        stays causal, because position `i * block_size` precedes every other position in
        block `i`, and the block's weights come from that same leading token.

        Only blocks complete in the past compete. That keeps a top-k slot from being spent
        on a block the local window forces open anyway, and keeps the diagonal block's
        summary -- the one place a key pool holds tokens from the query's own future --
        out of the decision entirely.
        """
        batch = queries.shape[0]
        summary = index_keys.view(
            batch, blocks, self.block_size, self.index_width).mean(2)
        leaders = queries.view(
            batch, blocks, self.block_size, self.index_heads, self.index_width)[:, :, 0]
        leader_weights = weights.view(
            batch, blocks, self.block_size, self.index_heads)[:, :, 0]
        gain = leader_weights.permute(0, 2, 1).unsqueeze(-1).float()

        if candidate_index is None:
            scores = torch.einsum("bqhd,bnd->bhqn", leaders.float(), summary.float())
            scores = (torch.relu(scores) * gain).sum(dim=1)
        else:
            # Level one narrowed the field, so score the narrowed field. Masking a full
            # `[q_blocks, kv_blocks]` product afterwards constrains the selection without
            # saving any of the work; gathering the candidate summaries first turns the
            # product from O(q * kv) into O(q * candidates), which is what the hierarchy
            # is for. Results scatter back to full width so everything downstream --
            # eligibility, the local window, top-k -- is unchanged.
            wide = candidate_index.shape[-1]
            gathered = torch.gather(
                summary.unsqueeze(1).expand(-1, blocks, -1, -1), 2,
                candidate_index.unsqueeze(-1).expand(-1, -1, -1, self.index_width))
            narrow = torch.einsum("bqhd,bqwd->bhqw", leaders.float(), gathered.float())
            narrow = (torch.relu(narrow) * gain).sum(dim=1)
            scores = torch.full((batch, blocks, blocks), float("-inf"),
                                dtype=narrow.dtype, device=narrow.device)
            scores.scatter_(-1, candidate_index, narrow)
            del gathered, narrow, wide

        rows = torch.arange(blocks, device=queries.device)
        offsets = rows.view(-1, 1) - rows.view(1, -1)
        return scores, offsets > self.local_blocks

    def _index_inputs(self, hidden_states, index_keys, queries, position_embeddings):
        """Index queries, index keys and per-head weights, roped and shaped."""
        batch, seq, _ = hidden_states.shape
        if self.isolated_indexer:
            # The first of DeepSeek's two cuts: whatever the indexer's own loss asks for,
            # it stops here rather than travelling back into the backbone. `queries` and
            # `index_keys` were projected before this point, so they are cut too.
            hidden_states = hidden_states.detach()
            queries = None if queries is None else queries.detach()
            index_keys = index_keys.detach()
        if queries is None:
            queries = self.index_q_proj(hidden_states)
        queries = queries.view(batch, seq, self.index_heads, self.index_width)
        if self.rope_index and position_embeddings is not None:
            # Only the query turns. Its trailing slice is the half that meets the cached
            # rotary key, which was turned at the position it was written; the leading
            # half meets a content key off the latent, which carries no position at all.
            queries = self._rope_index(
                queries.transpose(1, 2), position_embeddings).transpose(1, 2)
        if self.token_head_weights:
            # Softplus rather than the reference's raw projection: the head scores pass a
            # ReLU and are summed, and a negative weight would turn that sum into a
            # subtraction of evidence the ReLU already floored at zero.
            weights = torch.nn.functional.softplus(self.indexer_proj(hidden_states))
        else:
            weights = torch.nn.functional.softplus(self.index_weight)
            weights = weights.view(1, 1, -1).expand(batch, seq, -1)
        return queries, index_keys, weights

    def route(self, hidden_states, index_keys, queries=None, position_embeddings=None,
              candidates=None):
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
        if self.dense_routing:
            # Every causal block open and no router bias: what this model attends to when
            # the router is not deciding. It is the teacher a self-distillation needs,
            # because the model's own sparse attention is masked by the router being
            # trained, and a plain-MLA copy cannot be built -- the borrowing layers have
            # no down projection of their own to copy.
            rows = torch.arange(blocks, device=hidden_states.device)
            allowed = (rows.view(-1, 1) >= rows.view(1, -1)).unsqueeze(0).expand(
                batch, blocks, blocks).contiguous()
            self.last_candidates = None
            return allowed, None

        queries, index_keys, weights = self._index_inputs(
            hidden_states, index_keys, queries, position_embeddings)

        with torch.no_grad():
            scores, eligible = self._score_blocks(
                queries, index_keys, weights, blocks, candidates)
            rows = torch.arange(blocks, device=hidden_states.device)
            offsets = rows.view(-1, 1) - rows.view(1, -1)
            # The recent blocks always survive routing: a query that cannot see its own
            # immediate context produces gradients about the router, not the
            # architecture.
            local = (offsets >= 0) & (offsets <= self.local_blocks)
            # Level one already restricted the score to the candidate blocks, so
            # everything outside them is -inf here and needs no second mask.
            scores = scores.masked_fill(~eligible.unsqueeze(0), float("-inf"))

            keep = max(1, min(self.top_k // self.block_size, blocks))
            published = None
            if self.publishes_candidates:
                wide = max(keep, min(self.candidate_k // self.block_size, blocks))
                # The indices are the useful half: a later layer gathers those summaries
                # instead of scoring every block. The mask is kept beside them because it
                # is what a report or a test can read.
                self.last_candidate_index = scores.topk(wide, dim=-1).indices
                published = torch.zeros(batch, blocks, blocks, dtype=torch.bool,
                                        device=hidden_states.device)
                published.scatter_(-1, self.last_candidate_index, True)
                published |= ~eligible.unsqueeze(0)
            chosen = scores.topk(keep, dim=-1).indices
            allowed = torch.zeros(batch, blocks, blocks, dtype=torch.bool,
                                  device=hidden_states.device)
            allowed.scatter_(-1, chosen, True)
            # An early block has no eligible candidates, so its top-k over an all -inf
            # row returns arbitrary indices; this drops them. A row level one left short
            # is the same case and is dropped the same way.
            allowed &= eligible.unsqueeze(0)
            if candidates is not None:
                inside = torch.zeros_like(allowed)
                inside.scatter_(-1, candidates, True)
                allowed &= inside
            allowed |= local.unsqueeze(0)
            allowed &= (offsets >= 0).unsqueeze(0)

        # The heads collapse here. Summing head scores through a ReLU is what makes them
        # distinct, and the ReLU cannot be folded into a dot product; the linear part can,
        # and sum_h w_h * (Q_h . K) is exactly (sum_h w_h Q_h) . K.
        # Kept beside `last_allowed` rather than returned: every caller of `route` wants
        # the selection and the query, and only the publishing layer has a third thing.
        self.last_candidates = published
        effective = (queries * weights.unsqueeze(-1)).sum(dim=2)
        return allowed, effective

    @property
    def publishes_candidates(self) -> bool:
        return (self.candidate_k > 0 and self.mode != "reuse"
                and self.layer_idx == self.candidate_layer)

    def _inherited_candidates(self):
        """The candidate set from level one, for a layer that comes after the publisher.

        The publisher routes over the whole row -- narrowing it against its own output
        would make level one a no-op -- so it reads nothing here, and neither does any
        layer before it.
        """
        if self.candidate_k <= 0 or self.layer_idx <= self.candidate_layer:
            return None
        return self.bus.candidate_index

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

    def _blocked(self, seq, past_key_values):
        """Whether the block-sparse kernel can take this shape.

        FlexAttention's ``BlockMask`` groups queries as well as keys, so it needs whole
        query blocks: a one-token step computes zero of them and a sequence that does not
        divide loses its tail. The reference groups only the key axis -- `ggml_pool_2d`
        over positions with the token count untouched -- and takes its top-k per query
        token, which is why the same code serves a decode step there. `_forward_gathered`
        is that shape, and it takes every case this one cannot.
        """
        return (past_key_values is None and seq >= self.block_size
                and seq % self.block_size == 0)

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

        ``top_k`` counts tokens here, which is what it says it is. The blocked path floors
        it to ``top_k // block_size`` blocks because a block is the finest thing a
        ``BlockMask`` can express -- 2 blocks of 8 for a budget of 256 over a 1024-token
        window. Selecting positions spends the same budget at the granularity the
        reference uses.
        """
        scores = torch.einsum("bqhd,bkd->bhqk", index_queries.float(), index_keys.float())
        gain = weights.permute(0, 2, 1).unsqueeze(-1).float()
        scores = (torch.relu(scores) * gain).sum(dim=1)

        causal = kv_positions.view(1, 1, -1) <= q_positions.view(1, -1, 1)
        scores = scores.masked_fill(~causal, float("-inf"))
        keep = max(1, min(self.top_k, index_keys.shape[1]))
        allowed = torch.zeros_like(scores, dtype=torch.bool)
        allowed.scatter_(-1, scores.topk(keep, dim=-1).indices, True)
        # The recent window is not spent out of the top-k budget, the same way the blocked
        # path forces its local blocks open: a query that cannot see its own immediate
        # context produces gradients about the router rather than about the architecture.
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
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=mask, scale=self.scaling)
        attn_output = attn_output.transpose(1, 2).reshape(batch, seq, -1).contiguous()
        return self.o_proj(attn_output * torch.sigmoid(gate)), None

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
            hidden_states, queries = hidden_states.detach(), queries.detach()
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
        """The path that takes a decode step, and any length the block kernel refuses."""
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
                effective = (index_queries * weights.unsqueeze(-1)).sum(dim=2)
            self.bus.select(self.layer_idx, allowed)

        self.last_allowed = allowed.detach()
        self.last_token_routed = True
        self.last_candidates = None
        return self._attend_gathered(hidden_states, latent, rotary, allowed, effective,
                                     index_keys, position_embeddings, query)

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kwargs):
        batch, seq, _ = hidden_states.shape
        if self.bus is None:
            raise RuntimeError(
                "layer %d has no SparseIndexBus; the text model injects one after it "
                "builds its layers" % self.layer_idx)
        if not self._blocked(seq, past_key_values):
            return self._forward_gathered(hidden_states, position_embeddings,
                                          past_key_values, kwargs.get("cache_position"))
        self.last_token_routed = False

        if self.mode == "full":
            query, compressed, queries = self._project(hidden_states)
            latent, rotary = torch.split(compressed, [self.latent, self.rope_dim], dim=-1)
            latent = self.kv_a_norm(latent)
            rotary = self._rope_shared(rotary, position_embeddings)
            index_keys = self.index_keys_from(latent, rotary)
            allowed, effective = self.route(
                hidden_states, index_keys, queries, position_embeddings,
                self._inherited_candidates())
            self.bus.publish(self.layer_idx, index_keys, latent, rotary)
            self.bus.select(self.layer_idx, allowed)
            if self.last_candidates is not None:
                self.bus.candidates = self.last_candidates
                self.bus.candidate_index = self.last_candidate_index
        elif self.mode == "reindex":
            query, queries = self._project(hidden_states)
            index_keys, borrowed, rotary = self.bus.require_latent(
                self.latent_donor, self.layer_idx)
            latent = self.kv_adapt(borrowed)
            # New routing over borrowed keys: this layer's own view of what matters.
            allowed, effective = self.route(
                hidden_states, index_keys, queries, position_embeddings,
                self._inherited_candidates())
            self.bus.select(self.layer_idx, allowed)
            if self.last_candidates is not None:
                self.bus.candidates = self.last_candidates
                self.bus.candidate_index = self.last_candidate_index
        else:
            query, = self._project(hidden_states)
            _, borrowed, rotary = self.bus.require_latent(
                self.latent_donor, self.layer_idx)
            latent = self.kv_adapt(borrowed)
            allowed = self.bus.require_selection(self.topk_donor, self.layer_idx)
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
        if self.k_norm is not None:
            content_key = self.k_norm(content_key)

        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        cos, sin = position_embeddings
        # The query only. The shared slice arrives already turned, because the index keys
        # are built from it before this and both paths have to agree about when it
        # happens -- the gathered one turns it on the way into the cache, so this one
        # turns it on the way in too.
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)
        shared = rotary.unsqueeze(1).expand(input_shape[0], self.num_heads,
                                            input_shape[1], self.rope_dim)
        key_states = torch.cat([shared, content_key.transpose(1, 2)], dim=-1)
        value_states = value_states.transpose(1, 2)

        # FlexAttention compiles the block mask into a fused kernel that skips whole
        # blocks. DeepSeek's own sparse kernels are SM90/SM100 and this card is SM86, so
        # borrowing them is not available; this reaches the same place through Triton.
        # Measured against the alternatives at this shape: 5.51 ms here, 10.33 ms for
        # dense SDPA, 17.39 ms for SDPA with a token-level boolean mask.
        if effective is not None:
            heads, length = self.num_heads, input_shape[1]
            # `effective` came out of `route` built from *roped* index queries, so the key
            # it is paired with has to be roped too. The rotation is orthogonal: applied
            # to both sides it leaves the dot product alone, applied to one side it
            # silently changes the bias the router is trained through into something the
            # selection never used.
            if self.rope_index and position_embeddings is not None:
                index_keys = self._rope_index(index_keys, position_embeddings)
            query_extra, key_extra = self.router_columns(effective, index_keys)
            query_states = torch.cat(
                [query_states, query_extra.expand(-1, heads, length, -1)], dim=-1)
            key_states = torch.cat(
                [key_states, key_extra.expand(-1, heads, length, -1)], dim=-1)

        attn_output = _flex()(
            query_states, key_states, value_states, block_mask=mask, scale=self.scaling,
            kernel_options=self._kernel_options(query_states.shape[-1],
                                                query_states.element_size()))

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output), None

    def _kernel_options(self, width, element_size=2):
        """Tile sizes that fit this card's shared memory at this head width.

        FlexAttention rounds the key/query head up to a power of two before it allocates
        anything -- `QK_HEAD_DIM_ROUNDED = next_power_of_two(...)` in Inductor's flex
        common -- and then picks its tile from a table keyed on the *unrounded* width,
        with no check that the result fits the device. Past that table's last entry at
        256 it drops to a fixed 64x32 with three stages. Neither path has a fallback: the
        config that does not fit is dropped, the choice list empties, and the compile ends
        in "No valid triton configs" rather than in a smaller tile.

        So pick the tile here instead. The rounding is the larger of the two terms: the
        router's `index_dim` columns ride in `width`, and 256 + 64 rounds to 512, so every
        query and key tile doubles to carry 64 columns of content. `num_stages` then
        multiplies the key and value buffers that the rounding has already doubled.

        Measured on SM86, which offers 101376 bytes, at the real student's geometry of 8
        heads and 1024 positions:

            width 256   default 32x64x3 asks 151552   this 16x32x3   0.703 ms
            width 320   default 64x32x3 asks 167936   this 16x32x1   0.821 ms

        Both defaults are over the limit, including the one at 256, which is the dense
        path and carries no router columns at all. Re-measure the surface with
        `scratch/dense_gr/flex_tiles.py` if the head or the index width changes.

        A tile holds elements and shared memory holds bytes, so float32 doubles every one
        of these and the same shape that fits in bfloat16 asks 149568. What decides a tile
        is therefore the rounded head in *bytes*, and `element_size` is a parameter rather
        than an assumption about the arm's dtype because the indexer's teacher runs
        float32 -- and a self-distillation runs the student's own attention as that
        teacher, so those layers reach this in float32 too.

        The backward has a second table with the same fault and an inverted twist. On SM86
        a head *wider* than 256 gets `FlexBwDConfig(16, 16, 16, 16, 1, 4)`, which fits,
        and a head at exactly 256 gets 64x64x64x64 with two stages, which asks 168960. So
        a layer carrying the router's columns is saved by them, and a reuse layer, which
        has none, is the one that cannot compile. Forward-only work never sees it: the
        conversion and every evaluation run under `no_grad`.

            width 256   forward 16x32x3  0.703 ms   backward block 32 x2  4.428 ms
            width 320   forward 16x32x1  0.821 ms   backward block 16 x3  5.682 ms

        Both halves are named with their prefix. Inductor strips `fwd_` in the forward
        lowering and drops `bwd_`, and the reverse in the backward -- but a bare key
        survives into both, so a plain `num_stages` silently sets the backward's as well.

        The bfloat16 rows above are measured. The float32 ones are the same rule read at
        twice the span, which the byte formula says fits and `flex_tiles.py` is what would
        confirm.
        """
        # The toy widths land on the tuned entries of those tables and fit as they stand.
        # Leaving them alone keeps their timings comparable to what is already measured.
        span = (1 << (width - 1).bit_length()) * element_size
        if span <= 256:
            return None
        if span <= 512:
            return {"fwd_BLOCK_M": 16, "fwd_BLOCK_N": 32, "fwd_num_stages": 3,
                    "bwd_BLOCK_M1": 32, "bwd_BLOCK_N1": 32, "bwd_BLOCK_M2": 32,
                    "bwd_BLOCK_N2": 32, "bwd_num_stages": 2}
        if span <= 1024:
            return {"fwd_BLOCK_M": 16, "fwd_BLOCK_N": 32, "fwd_num_stages": 1,
                    "bwd_BLOCK_M1": 16, "bwd_BLOCK_N1": 16, "bwd_BLOCK_M2": 16,
                    "bwd_BLOCK_N2": 16, "bwd_num_stages": 3}
        return {"fwd_BLOCK_M": 16, "fwd_BLOCK_N": 16, "fwd_num_stages": 1,
                "bwd_BLOCK_M1": 16, "bwd_BLOCK_N1": 16, "bwd_BLOCK_M2": 16,
                "bwd_BLOCK_N2": 16, "bwd_num_stages": 1}

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
        # The gathered path selects positions rather than blocks, so the unit the window
        # is measured in changes with it. Everything below is the same arithmetic over
        # whichever unit the last forward routed in.
        near = self.local_window if self.last_token_routed else self.local_blocks
        columns = torch.arange(blocks, device=allowed.device)
        query_rows = (columns if not self.last_token_routed
                      else torch.arange(blocks - allowed.shape[-2], blocks,
                                        device=allowed.device))
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
            "unit": "positions" if self.last_token_routed else "blocks",
            "blocks": blocks,
            "density": float(allowed.sum()) / float(reachable.expand_as(allowed).sum()),
            "selected": (float(chosen.sum()) / eligible_total) if eligible_total else 0.0,
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
