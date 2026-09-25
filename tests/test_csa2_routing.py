"""What CSA2 and MLA have to get right for a training arm to mean anything.

Three of these cover failures that a run would not report. A router whose parameters take
no gradient still produces a loss curve, so "the model trained" says nothing about whether
the routing trained with it. A local window that is claimed and not enforced only shows up
as slightly worse numbers. And a cache that holds the expanded per-head keys instead of the
latent is four times larger than the GQA it replaces while every printed figure says it is
smaller.
"""

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from distillkit.models import Qwen35WidenedForCausalLM
from distillkit.models.qwen35.csa2 import (Qwen35SparseLatentAttention,
                                            SparseIndexBus, dense_routing,
                                            isolated_indexer, recorded_attention,
                                            router_parameters)

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="FlexAttention compiles a Triton kernel")

BLOCK, HEADS, HEAD_DIM = 64, 2, 32
MODES = ["full", "reuse", "full", "reindex", "reuse"]


def tiny_config(**kwargs):
    """A ten-layer stack with the five CSA2 modes on its full-attention layers."""
    pairs = int(HEAD_DIM * 0.25) // 2
    config = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=HEADS * HEAD_DIM, intermediate_size=128,
        num_hidden_layers=10, num_attention_heads=HEADS, num_key_value_heads=1,
        head_dim=HEAD_DIM, linear_key_head_dim=HEAD_DIM, linear_value_head_dim=HEAD_DIM,
        linear_num_key_heads=1, linear_num_value_heads=HEADS, linear_conv_kernel_dim=4,
        full_attention_interval=2, tie_word_embeddings=True,
        max_position_embeddings=512, pad_token_id=0, eos_token_id=3, use_cache=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25, "mrope_interleaved": True,
                         "mrope_section": [pairs - 2 * (pairs // 3), pairs // 3,
                                           pairs // 3]},
    )
    config.residual_stream_enabled = True
    config.residual_stream_routing = "flash_next"
    config.residual_stream_num_branches = 2
    config.residual_stream_lowrank = 8
    config.residual_stream_sidecar = False
    config.sidecar_layer_index = 0
    config.mla_enabled = True
    config.mla_latent_dim = 32
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


@pytest.mark.parametrize("ratio,interval,full", [("1:1", 2, 5), ("3:1", 4, 2)])
def test_attention_ratio_places_the_layers(ratio, interval, full):
    """Both ratios are selectable, and depth buys a different number of full layers.

    `full_attention_interval` counts layers, not attention layers, so the same ten-layer
    stack has five full-attention layers at 1:1 and two at 3:1. A CSA2 mode list is one
    entry per full-attention layer, so a pattern written for one ratio needs a different
    depth at the other.
    """
    import sys
    sys.path.insert(0, "scratch/dense_gr")
    from benchmark import ATTENTION_RATIOS, build, full_attention_layers

    assert ATTENTION_RATIOS[ratio] == interval
    config = build(256, 10, 64, ratio=ratio)
    placed = [i for i, kind in enumerate(config.layer_types) if "linear" not in str(kind)]
    assert placed == full_attention_layers(10, ratio)
    assert len(placed) == full
    # The pattern this project runs needs one entry per full-attention layer, so keeping
    # five of them at 3:1 means twenty layers rather than ten.
    assert len(build(256, len(MODES) * interval, 64, ratio=ratio).layer_types) \
        == len(MODES) * interval

    with pytest.raises(ValueError, match="unknown attention ratio"):
        build(256, 10, 64, ratio="2:1")
    with pytest.raises(ValueError, match="at least"):
        build(256, interval - 1, 64, ratio=ratio)


def test_csa2_without_mla_is_refused():
    """CSA2 installs inside the MLA branch, so the flag alone silently does nothing.

    Without this the run builds ordinary attention, trains, and reports a CSA2
    architecture it never had.
    """
    config = csa2_config()
    config.mla_enabled = False
    with pytest.raises(ValueError, match="csa2_enabled requires mla_enabled"):
        Qwen35WidenedForCausalLM(config)


@pytest.mark.parametrize("blend,active", [(0.0, False), (1.0, True)])
def test_gated_residual_is_inert_until_blend_is_set(blend, active):
    """`blend` is what decides whether the routing exists, and zero means it does not.

    `HyperConnection.read` short-circuits the donor arithmetic at blend 0, so every
    routing parameter takes exactly zero gradient and the model is a plain pre-norm stack
    carrying weights that never move. An arm that means to exercise GR has to say so, and
    an arm that does not should know it is not.
    """
    import sys
    sys.path.insert(0, "scratch/dense_gr")
    from benchmark import build, variant_tag

    assert variant_tag("1:1", blend) == ("r1-1-gr" if active else "r1-1-nogr")
    config = build(128, 4, 256, head_dim=32, ratio="1:1", blend=blend,
                   attn_implementation="eager")
    assert config.residual_stream_blend == blend
    torch.manual_seed(0)
    model = Qwen35WidenedForCausalLM(config).float()
    model.train()
    tokens = torch.randint(0, 256, (2, 16))
    model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                use_cache=False).last_hidden_state.pow(2).mean().backward()

    routing = [(name, parameter) for name, parameter in model.named_parameters()
               if any(part in name for part in
                      ("W_down", "W_up", "W_write", "branch_gain_delta"))]
    assert routing
    moved = [name for name, parameter in routing
             if parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)]
    assert bool(moved) is active, (
        "blend %.1f: %d of %d routing parameters took gradient"
        % (blend, len(moved), len(routing)))


def csa2_config(**kwargs):
    defaults = dict(csa2_enabled=True, csa2_modes=MODES, csa2_top_k=BLOCK,
                    csa2_local_window=8, csa2_block_size=BLOCK, csa2_index_dim=16,
                    csa2_index_heads=2)
    defaults.update(kwargs)
    return tiny_config(**defaults)


def bare_layer(mode="full", **kwargs):
    """One attention module, off any model, for testing routing on the CPU."""
    layer = Qwen35SparseLatentAttention(csa2_config(**kwargs), 0, mode)
    layer.bus = None
    return layer


def index_keys_for(layer, hidden):
    """Index keys the way the model builds them: off the latent and the rotary key.

    They used to be a projection of the hidden state and cached beside the latent. They
    are a projection of the *latent* now, with the decoupled rotary key as their position
    half, so a test that wants them has to go through the layer's own compression rather
    than reach for the hidden state directly. Standing in a random latent instead would
    cut the keys loose from the content they are supposed to describe, and a router asked
    to find something in that content could not.
    """
    compressed = layer.kv_a_proj(hidden)
    latent, rotary = torch.split(compressed, [layer.latent, layer.rope_dim], dim=-1)
    return layer.index_keys_from(layer.kv_a_norm(latent), rotary)


def select(layer, hidden):
    """The token path's selection for a bare layer, expanded to `[batch, query, key]`."""
    keys = index_keys_for(layer, hidden)
    queries, weights = layer._index_queries(hidden, None, None)
    positions, valid = layer._select_positions(queries, keys, weights)
    seq = hidden.shape[1]
    counts = torch.zeros(hidden.shape[0], seq, seq, dtype=torch.int16)
    counts.scatter_add_(-1, positions, valid.to(torch.int16))
    return counts > 0


@cuda
def test_indexer_parameters_receive_gradient():
    """Top-k is discrete; without the score bias the indexer never learns anything.

    This is the test that fails on the selection-only router: every one of these comes
    back ``grad=None``, because indices, a scatter and a boolean mask break the graph.
    """
    torch.manual_seed(0)
    model = Qwen35WidenedForCausalLM(csa2_config()).cuda()
    model.train()
    tokens = torch.randint(0, 64, (2, 4 * BLOCK), device="cuda")
    out = model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                      use_cache=False).last_hidden_state
    out.float().pow(2).mean().backward()

    checked = 0
    for name, parameter in model.named_parameters():
        if not any(part in name for part in
                   ("index_q_proj", "index_k_proj", "index_weight", "indexer_proj",
                    "index_gate")):
            continue
        checked += 1
        assert parameter.grad is not None, "%s took no gradient" % name
        assert torch.isfinite(parameter.grad).all(), "%s gradient is not finite" % name
        assert parameter.grad.abs().sum() > 0, "%s gradient is all zero" % name
    # Two full layers own queries, keys, head weights and a gate; the reindex layer owns
    # everything but the keys. The head weights are one tensor either way -- a projection
    # under the default, a shared vector under `csa2_token_head_weights=False`.
    assert checked == 11


@pytest.mark.parametrize("window", [1, 8, BLOCK, BLOCK + 1, 3 * BLOCK])
def test_local_window_is_always_routed(window):
    """Every query keeps its last ``local_window`` positions whatever the router scores."""
    torch.manual_seed(0)
    layer = bare_layer(csa2_local_window=window)
    hidden = torch.randn(2, 6 * BLOCK, layer.config.hidden_size)
    allowed = select(layer, hidden)
    rows = torch.arange(hidden.shape[1])
    offsets = rows.view(-1, 1) - rows.view(1, -1)
    window_mask = (offsets >= 0) & (offsets < window)
    assert bool(allowed[:, window_mask].all()), "window %d not read" % window


@pytest.mark.parametrize("scale", [1.0, 8.0])
@pytest.mark.parametrize("window", [0, 8])
def test_routing_does_not_depend_on_a_token_s_own_future(window, scale):
    """Mutating one token must not change what any earlier query may read.

    One routing decision per query token, from that token's own index query against keys
    at or before it. The old block routing made one decision per block and had to take it
    from the block's leading query to stay causal; this checks the property directly.
    """
    torch.manual_seed(0)
    length = 10 * BLOCK
    layer = bare_layer(csa2_local_window=window, csa2_top_k=2 * BLOCK)
    width = layer.config.hidden_size
    for _ in range(10):
        victim = int(torch.randint(1, length, ()))
        hidden = torch.randn(1, length, width)
        before = select(layer, hidden)
        changed = hidden.clone()
        changed[:, victim] += scale * torch.randn(width)
        after = select(layer, changed)
        assert torch.equal(before[0, :victim], after[0, :victim]), victim


@cuda
def test_routing_report_describes_the_last_forward():
    """Density alone cannot spot a collapsed router, and the entropy has to be honest.

    A router that has fallen back onto the local window still reports a healthy density,
    because the window is open regardless of score -- so `selected` measures only the
    positions top-k was free to choose. And entropy normalized by the positions actually
    used would score an even split over two of them as a perfect 1.0, which is the
    collapse it exists to catch, so it is normalized by the positions available instead.
    """
    from distillkit.models.qwen35.csa2 import routing_report

    torch.manual_seed(0)
    model = Qwen35WidenedForCausalLM(csa2_config(csa2_top_k=2 * BLOCK,
                                                 csa2_router_bias=False)).cuda()
    model.eval()
    assert routing_report(model) == [], "nothing has run yet"

    tokens = torch.randint(0, 64, (4, 8 * BLOCK), device="cuda")
    with torch.no_grad():
        model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                    use_cache=False)
    rows = routing_report(model)
    assert [r["mode"] for r in rows] == MODES
    for row in rows:
        assert row["blocks"] == 8 * BLOCK and row["unit"] == "positions"
        assert 0.0 < row["density"] < 1.0, row
        assert 0.0 < row["selected"] < 1.0, row
        assert 0.0 < row["entropy"] <= 1.0, row

    # A router pinned to one key position must read as collapsed, not as healthy density.
    layer = next(m for m in model.modules()
                 if isinstance(m, Qwen35SparseLatentAttention))
    pinned = torch.zeros_like(layer.last_allowed)
    rows_index = torch.arange(pinned.shape[1], device=pinned.device)
    pinned[:, :, 0] = True
    pinned |= (rows_index.view(-1, 1) == rows_index.view(1, -1)).unsqueeze(0)
    layer.last_allowed = pinned & (rows_index.view(-1, 1)
                                   >= rows_index.view(1, -1)).unsqueeze(0)
    collapsed = layer.routing_statistics()
    assert collapsed["entropy"] == pytest.approx(0.0), collapsed


@pytest.mark.parametrize("mode,pieces", [("full", 3), ("reindex", 2), ("reuse", 1)])
def test_fused_projection_matches_separate_ones(mode, pieces):
    """One multiply over the stream must give what the separate multiplies gave.

    A Full layer projects the query, the latent and the index queries from the same
    `[batch, tokens, hidden]` tensor, and two of the three are narrow enough that reading
    that tensor dominates. Sharing the multiply is only worth doing if the split puts
    every piece back exactly where it was -- a wrong order or width would train perfectly
    well and mean something else.

    The index *keys* used to be a fourth piece here. They come off the latent now, which
    is 384 wide against the hidden state's 2048, so they are no longer reading the tensor
    this fusion exists to read once.
    """
    torch.manual_seed(0)
    layer = bare_layer(mode) if mode != "full" else bare_layer()
    if mode != "full":
        layer = Qwen35SparseLatentAttention(csa2_config(), 0, mode)
    hidden = torch.randn(2, 4 * BLOCK, layer.config.hidden_size)

    projected = layer._project(hidden)
    assert len(projected) == pieces
    expected = [layer.q_proj]
    if mode == "full":
        expected.append(layer.kv_a_proj)
    if mode in ("full", "reindex"):
        expected.append(layer.index_q_proj)
    for part, module in zip(projected, expected):
        assert torch.allclose(part, module(hidden), atol=1e-6, rtol=1e-6)
        assert part.shape[-1] == module.out_features


def test_routing_stays_causal_and_sparse():
    torch.manual_seed(0)
    layer = bare_layer(csa2_local_window=0, csa2_top_k=BLOCK)
    hidden = torch.randn(2, 8 * BLOCK, layer.config.hidden_size)
    allowed = select(layer, hidden)
    rows = torch.arange(hidden.shape[1])
    causal = rows.view(-1, 1) >= rows.view(1, -1)
    assert not bool((allowed & ~causal.unsqueeze(0)).any())
    # Something has to be dropped, or the sparsity is decorative.
    assert allowed.sum() < causal.sum() * allowed.shape[0]


def test_router_columns_carry_a_bounded_cosine():
    """The appended columns must reproduce ``index_gate * cos`` once attention scales.

    This is the router bias's whole gradient path, and it is invisible in any output: get
    the scaling wrong and the router still trains, against a logit shift of the wrong size.
    """
    torch.manual_seed(0)
    layer = bare_layer()
    with torch.no_grad():
        layer.index_gate.fill_(0.7)
    hidden = torch.randn(2, 2 * BLOCK, layer.config.hidden_size)
    keys = index_keys_for(layer, hidden)
    queries, weights = layer._index_queries(hidden, None, None)
    effective = (queries * weights.unsqueeze(-1)).sum(dim=2)

    query_extra, key_extra = layer.router_columns(effective, keys)
    contributed = (query_extra @ key_extra.transpose(-1, -2)) * layer.scaling
    cosine = torch.nn.functional.cosine_similarity(
        effective.unsqueeze(2), keys.unsqueeze(1), dim=-1)
    assert torch.allclose(contributed.squeeze(1), 0.7 * cosine, atol=1e-5)
    assert contributed.abs().max() <= 0.7 + 1e-5


def test_bus_lives_on_the_model_not_the_config():
    """Two models from one config must not share routing state, and the config stays JSON."""
    config = csa2_config()
    first = Qwen35WidenedForCausalLM(config)
    second = Qwen35WidenedForCausalLM(config)
    assert not hasattr(config, "csa2_bus")
    assert first.model.csa2_bus is not second.model.csa2_bus
    for model in (first, second):
        for layer in model.model.layers:
            attention = getattr(layer, "self_attn", None)
            if isinstance(attention, Qwen35SparseLatentAttention):
                assert attention.bus is model.model.csa2_bus
    config.to_json_string()


@pytest.mark.parametrize("length", [1, BLOCK + 1, BLOCK])
def test_every_length_routes_per_token(length):
    """A ragged length and a decode-sized step are ordinary cases, not refusals."""
    layer = bare_layer()
    layer.bus = SparseIndexBus()
    hidden = torch.randn(1, length, layer.config.hidden_size)
    width = int(layer.head_dim * 0.25)
    position = (torch.ones(1, length, width), torch.zeros(1, length, width))
    with torch.no_grad():
        out, _ = layer.forward(hidden, position_embeddings=position)
    assert out.shape == hidden.shape
    assert layer.last_token_routed
    # Selected positions, not blocks: the granularity the reference uses.
    assert layer.last_allowed.shape == (1, length, length)


def test_decoding_through_the_cache_matches_one_whole_forward():
    """The property the decode path exists for, and the only one that makes it useful.

    A cache that is the right size and the wrong contents shows up nowhere else: every
    evaluation runs ``use_cache=False``, so nothing else exercises this. The router's
    logit bias is the easiest thing to drop here -- it is carried by extra query and key
    columns rather than by the mask -- and dropping it would leave a decoded token quietly
    disagreeing with the same token read whole.
    """
    torch.manual_seed(0)
    model = Qwen35WidenedForCausalLM(csa2_config()).eval()
    model.config.use_cache = True
    # A length the block kernel would not take, so both sides of the comparison run the
    # gathered path and the cache is the only thing that differs. The blocked path needs
    # a compiler for CPU Inductor, which is a property of the environment, not the claim.
    length = BLOCK + 1
    tokens = torch.randint(1, 64, (1, length))

    with torch.no_grad():
        whole = model(input_ids=tokens, use_cache=False).logits
        out = model(input_ids=tokens[:, :-4], use_cache=True)
        cache, step = out.past_key_values, [out.logits[:, -1]]
        for index in range(length - 4, length - 1):
            out = model(input_ids=tokens[:, index:index + 1],
                        past_key_values=cache, use_cache=True)
            cache, _ = out.past_key_values, step.append(out.logits[:, -1])

    decoded = torch.cat(step)
    reference = whole[0, -5:-1]
    assert decoded.shape == reference.shape
    assert torch.allclose(decoded, reference, atol=2e-3), \
        "worst position differs by %.5f" % (decoded - reference).abs().max()


def test_absorbing_the_up_projection_computes_the_same_attention():
    """`q . (W_k c) = (W_k^T q) . c` is an identity, so the two paths must agree.

    Not approximately: the expanding path and the absorbed one are the same function
    written twice, and anything that breaks the identity -- a norm between the latent and
    the key, the router's columns dropped, the scale applied in the wrong place -- shows
    up here and nowhere else, because absorption only runs when a cache is being read and
    every other test runs whole sequences.
    """
    torch.manual_seed(0)
    model = Qwen35WidenedForCausalLM(csa2_config()).eval()
    model.config.use_cache = True
    length = BLOCK + 1
    tokens = torch.randint(1, 64, (1, length))

    layers = [l.self_attn for l in model.model.layers
              if isinstance(getattr(l, "self_attn", None), Qwen35SparseLatentAttention)]
    assert all(a.k_norm is None for a in layers), "absorption needs the key norm gone"

    with torch.no_grad():
        out = model(input_ids=tokens[:, :-1], use_cache=True)
        cache = out.past_key_values
        step = tokens[:, -1:]
        absorbed = model(input_ids=step, past_key_values=cache,
                         use_cache=True).logits[:, -1]

    # The same step with absorption refused, by giving the layers a key norm that is the
    # identity: it changes nothing arithmetically and sends the forward down the other
    # branch.
    torch.manual_seed(0)
    plain = Qwen35WidenedForCausalLM(csa2_config()).eval()
    plain.load_state_dict(model.state_dict())
    plain.config.use_cache = True
    for attention in plain.model.layers:
        inner = getattr(attention, "self_attn", None)
        if isinstance(inner, Qwen35SparseLatentAttention):
            inner.k_norm = torch.nn.Identity()
    with torch.no_grad():
        out = plain(input_ids=tokens[:, :-1], use_cache=True)
        expanded = plain(input_ids=step, past_key_values=out.past_key_values,
                         use_cache=True).logits[:, -1]

    gap = (absorbed - expanded).abs().max()
    assert gap < 2e-2, "absorbed and expanded attention differ by %.4g" % gap


def test_a_borrowing_layer_writes_no_cache_and_a_full_one_holds_the_index_keys():
    """Where the second halving comes from, asserted on the cache rather than a formula."""
    model = Qwen35WidenedForCausalLM(csa2_config()).eval()
    model.config.use_cache = True
    with torch.no_grad():
        out = model(input_ids=torch.randint(1, 64, (1, BLOCK + 1)), use_cache=True)

    layers = {i: layer.self_attn for i, layer in enumerate(model.model.layers)
              if isinstance(getattr(layer, "self_attn", None), Qwen35SparseLatentAttention)}
    full = [i for i, a in layers.items() if a.mode == "full"]
    borrowing = [i for i, a in layers.items() if a.mode != "full"]
    assert full and borrowing

    for index in borrowing:
        assert layers[index].cached_numbers_per_token() == 0
    owner = layers[full[0]]
    # The latent alone in the key slot, the rotary slice in the value. The index keys are
    # rebuilt from the two rather than stored, which is what stopped a CSA2 layer caching
    # more than the plain MLA it compresses.
    entry = out.past_key_values.layers[full[0]]
    assert entry.keys.shape[-1] == owner.latent
    assert entry.values.shape[-1] == owner.rope_dim
    assert owner.cached_numbers_per_token() == owner.latent + owner.rope_dim


@cuda
def test_isolating_the_indexer_cuts_the_gradient_both_ways():
    """DeepSeek's sparse stage, asserted as the two things it actually claims.

    "The training signal of the indexer is from only L_I, while the optimization of the
    main model is according to only the language modeling loss." So a language-modeling
    loss must reach the backbone and not the indexer, which is the half this fork can get
    wrong in a way the reference cannot: its selection is a mask with no path from the
    loss to the indexer, and this one folds the index score into the logits through extra
    columns that exist precisely to carry that gradient.
    """
    torch.manual_seed(0)
    model = Qwen35WidenedForCausalLM(csa2_config()).cuda()
    tokens = torch.randint(1, 64, (1, 2 * BLOCK)).cuda()
    router = dict(router_parameters(model))
    backbone = {n: p for n, p in model.named_parameters()
                if n not in router and "self_attn" in n}
    assert router and backbone

    with isolated_indexer(model):
        model(input_ids=tokens, labels=tokens, use_cache=False).loss.backward()

    moved = [n for n, p in router.items() if p.grad is not None and p.grad.abs().sum() > 0]
    assert not moved, "the language model reached the indexer through %s" % moved
    reached = [n for n, p in backbone.items()
               if p.grad is not None and p.grad.abs().sum() > 0]
    assert reached, "isolating the indexer also stopped the backbone training"


@cuda
def test_without_isolation_the_indexer_still_takes_the_loss_gradient():
    """The contrast that makes the previous test mean something.

    Without it, a detach that silently cut everything, or a router that never took the
    loss gradient in the first place, would both look like success.
    """
    torch.manual_seed(0)
    model = Qwen35WidenedForCausalLM(csa2_config()).cuda()
    tokens = torch.randint(1, 64, (1, 2 * BLOCK)).cuda()
    model(input_ids=tokens, labels=tokens, use_cache=False).loss.backward()
    moved = [n for n, p in router_parameters(model)
             if p.grad is not None and p.grad.abs().sum() > 0]
    assert moved, "the router takes no loss gradient even unisolated; the contrast is void"


def test_the_attention_target_is_a_distribution_over_reachable_positions():
    """What `recorded_attention` hands the indexer has to be the thing it can predict.

    DeepSeek aligns the indexer to "the main attention distribution", summed over heads
    and normalized. If it were not normalized, or leaked onto positions the query cannot
    reach, the KL would be pulling the router toward something no selection could match.
    """
    torch.manual_seed(0)
    model = Qwen35WidenedForCausalLM(csa2_config()).eval()
    tokens = torch.randint(1, 64, (1, BLOCK + 1))
    layers = [l.self_attn for l in model.model.layers
              if isinstance(getattr(l, "self_attn", None), Qwen35SparseLatentAttention)
              and l.self_attn.mode != "reuse"]

    with torch.no_grad(), dense_routing(model), recorded_attention(model):
        model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                    use_cache=False)
        recorded = [attention.last_attention for attention in layers]

    assert all(r is not None for r in recorded), "no layer recorded its attention"
    for target in recorded:
        rows = target.sum(-1)
        assert torch.allclose(rows, torch.ones_like(rows), atol=1e-4)
        length = target.shape[-1]
        positions = torch.arange(length)
        future = positions.view(1, -1) > positions.view(-1, 1)
        assert float(target.squeeze(0)[future].abs().max()) == 0.0, \
            "the target puts weight on positions the query cannot reach"


@cuda
def test_the_warm_up_objective_reaches_every_indexer_parameter():
    """The point of the whole stage: a signal that does not need the logit bias.

    A discrete top-k carries no gradient, which is why this fork folds the index score
    into the attention logits at all. If the KL cannot move every indexer tensor on its
    own then removing those columns would leave a router that cannot train, and the
    reference's two-stage recipe would not port.
    """
    import sys
    sys.path.insert(0, "scratch/dense_gr")
    from indexer_kl import routing_layers, step

    torch.manual_seed(0)
    model = Qwen35WidenedForCausalLM(csa2_config()).cuda().float()
    tokens = torch.randint(1, 64, (1, 2 * BLOCK)).cuda()
    trained = dict(router_parameters(model))
    assert trained

    step(model, routing_layers(model), tokens).backward()
    missing = [name for name, parameter in trained.items()
               if parameter.grad is None or parameter.grad.abs().sum() == 0]
    # `index_gate` scales the logit bias and appears in no score, so the KL cannot reach
    # it -- and does not need to, since the reference has no such scalar.
    missing = [name for name in missing if "index_gate" not in name]
    assert not missing, "the KL never reached %s" % missing


def test_padding_is_refused():
    """Padding is not representable per block, so it is refused rather than attended.

    Checkpointing used to be refused alongside it, because the bus was written in forward
    order and a recompute reads it out of order. The bus is keyed by publisher now and
    `test_csa2_checkpoints_to_the_same_gradient_it_computes_without` pins the result.
    """
    model = Qwen35WidenedForCausalLM(csa2_config())
    tokens = torch.randint(1, 64, (2, BLOCK))
    mask = torch.ones_like(tokens)
    mask[0, -3:] = 0
    with pytest.raises(ValueError, match="cannot represent padding"):
        model.model(input_ids=tokens, attention_mask=mask, use_cache=False)


@pytest.mark.parametrize("mla", [False, True])
@pytest.mark.parametrize("chunk", [1, 4])
def test_chunked_cache_continuation_matches_the_whole_sequence(mla, chunk):
    """Feeding a sequence in pieces with a cache must reproduce feeding it at once.

    The trap is the 2D mask's width. `create_causal_mask` documents `attention_mask` as
    ``(batch, seen_tokens + query_length)``, not query length: hand it a chunk-width mask
    and every query in a continued chunk sees only the cached keys and not its own chunk,
    including itself. That is silent -- the shapes all work out -- and it looks exactly
    like a broken cache.
    """
    torch.manual_seed(0)
    config = tiny_config(mla_enabled=mla)
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    config._attn_implementation = "eager"
    model = Qwen35WidenedForCausalLM(config).double().eval()
    tokens = torch.randint(1, 64, (1, 8))

    def mask(width):
        return torch.ones(1, width, dtype=torch.long)

    with torch.no_grad():
        whole = model.model(input_ids=tokens, attention_mask=mask(8),
                            use_cache=False).last_hidden_state
        cache, pieces = None, []
        for start in range(0, 8, chunk):
            out = model.model(input_ids=tokens[:, start:start + chunk],
                              attention_mask=mask(start + chunk),
                              past_key_values=cache, use_cache=True)
            cache, _ = out.past_key_values, pieces.append(out.last_hidden_state)
    assert torch.allclose(whole, torch.cat(pieces, dim=1), atol=1e-12, rtol=1e-12)


def test_mla_caches_the_latent_not_the_expanded_heads():
    """The cache holds ``latent + rope_dim``, and replaying it reproduces the dense run.

    Handing `update` the assembled per-head keys caches ``2 * num_heads * head_dim``
    instead -- four times the GQA this replaces, while `cached_numbers_per_token` keeps
    reporting the latent width.
    """
    from transformers.cache_utils import DynamicCache

    torch.manual_seed(0)
    config = tiny_config()
    config._attn_implementation = "eager"
    model = Qwen35WidenedForCausalLM(config).eval()
    attention = next(layer.self_attn for layer in model.model.layers
                     if hasattr(getattr(layer, "self_attn", None), "kv_a_proj"))

    seq, split = 8, 4
    hidden = torch.randn(1, seq, config.hidden_size)
    position_ids = torch.arange(seq).view(1, 1, -1).expand(3, 1, -1)
    cos, sin = model.model.rotary_emb(hidden, position_ids)

    def causal(rows, columns):
        grid = torch.arange(columns - rows, columns).view(-1, 1) >= torch.arange(columns)
        return torch.zeros(1, 1, rows, columns).masked_fill(~grid, float("-inf"))

    with torch.no_grad():
        whole, _ = attention(hidden, position_embeddings=(cos, sin),
                             attention_mask=causal(seq, seq))
        cache, pieces = DynamicCache(config=config), []
        for start in (0, split):
            stop = start + split
            pieces.append(attention(
                hidden[:, start:stop],
                position_embeddings=(cos[:, start:stop], sin[:, start:stop]),
                attention_mask=causal(split, stop), past_key_values=cache)[0])
    assert torch.allclose(whole, torch.cat(pieces, dim=1), atol=1e-5, rtol=1e-5)

    stored = cache.layers[attention.layer_idx]
    keys, values = stored.keys, stored.values
    width = keys.shape[-1] + values.shape[-1]
    assert keys.shape[2] == values.shape[2] == seq
    rope_dim = int(HEAD_DIM * 0.25)
    assert width == attention.cached_numbers_per_token() == config.mla_latent_dim + rope_dim
    # The GQA it replaces caches a key and a value per key/VALUE head, not per query
    # head: a bound taken from num_attention_heads would be twice as loose here and
    # would pass a cache that had regressed past the thing it replaces.
    assert width < 2 * config.num_key_value_heads * HEAD_DIM




