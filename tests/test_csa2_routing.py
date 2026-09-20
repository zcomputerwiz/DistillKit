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
                                            SparseIndexBus, isolated_indexer,
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


@cuda
def test_router_learns_which_blocks_to_select():
    """Gradient is not the claim; better top-k is. So measure the top-k, not the gradient.

    The surrogate that carries the gradient is not the score that makes the selection --
    selection uses block-pooled per-head ReLU scores under a max, the surrogate a
    normalized per-token cosine over the head-weighted sum. Nonzero gradient therefore
    shows the parameters move, not that the decisions improve, and those are different
    claims.

    The task is a pointer. Every key block carries a signature and every token carries a
    copy of one block's signature, which is the block it should route to. Only the last
    query block is scored, its target is drawn strictly outside the blocks it gets for
    free, teacher and student are given identical local access so the target is the only
    difference between them, and the rate is measured on batches never trained on -- so
    the number cannot be inflated by forced blocks, impossible future targets, or
    memorizing a fixed batch.
    """
    blocks, batch = 8, 32
    seq = blocks * BLOCK
    torch.manual_seed(0)
    config = csa2_config(csa2_local_window=BLOCK, csa2_top_k=BLOCK, csa2_index_dim=16,
                         csa2_index_heads=2)
    layer = Qwen35SparseLatentAttention(config, 0, "full").cuda().float()
    layer.bus = SparseIndexBus()

    last = blocks - 1
    # Candidates the last block must actually route to: strictly in its past, and outside
    # the window it is handed regardless of score.
    choices = last - layer.local_blocks
    signature = torch.randn(blocks, config.hidden_size, device="cuda")
    signature /= signature.norm(dim=-1, keepdim=True)
    rows = torch.arange(batch, device="cuda")

    def sample(seed):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        hidden = torch.randn(batch, seq, config.hidden_size, device="cuda",
                             generator=generator)
        hidden += 4.0 * signature.repeat_interleave(BLOCK, 0).unsqueeze(0)
        target = torch.randint(0, choices, (batch,), device="cuda", generator=generator)
        # The pointer rides on the last block only, which is the block being scored.
        hidden[:, last * BLOCK:] += 2.0 * signature[target].unsqueeze(1)
        return hidden, target

    cos = torch.ones(batch, seq, int(layer.rope_dim), device="cuda")
    sin = torch.zeros_like(cos)

    def forward(hidden, target, force):
        keys = layer.index_k_proj(hidden)
        latent, rotary = torch.split(layer.kv_a_proj(hidden),
                                     [layer.latent, layer.rope_dim], dim=-1)
        latent = layer.kv_a_norm(latent)
        allowed, effective = layer.route(hidden, keys)
        if force:
            # Same local access as the student; the target is the only thing added, so
            # matching the teacher means routing there and nowhere else.
            offsets = torch.arange(blocks, device="cuda")
            offsets = offsets.view(-1, 1) - offsets.view(1, -1)
            allowed = ((offsets >= 0) & (offsets <= layer.local_blocks)).unsqueeze(0)
            allowed = allowed.expand(batch, blocks, blocks).clone()
            allowed[rows, last, target] = True
        return layer._attend(hidden, latent, rotary, layer.block_mask(allowed),
                             effective, keys, (cos, sin))[0]

    def selection_rate(seeds):
        """How often the last block's top-k finds the pointed-at block, on fresh data."""
        hits = []
        for seed in seeds:
            hidden, target = sample(seed)
            with torch.no_grad():
                allowed, _ = layer.route(hidden, layer.index_k_proj(hidden))
            hits.append(allowed[rows, last, target].float().mean().item())
        return sum(hits) / len(hits)

    indexer = [parameter for name, parameter in layer.named_parameters()
               if name.startswith(("index_q_proj", "index_k_proj", "index_weight",
                                   "index_gate"))]
    for parameter in layer.parameters():
        parameter.requires_grad_(False)
    for parameter in indexer:
        parameter.requires_grad_(True)

    held_out = [101, 102, 103, 104]
    before, opened = selection_rate(held_out), None
    optimizer = torch.optim.Adam(indexer, lr=1e-2)
    for step in range(300):
        hidden, target = sample(step)
        with torch.no_grad():
            teacher = forward(hidden, target, force=True)
        loss = (forward(hidden, target, force=False) - teacher)[:, last * BLOCK:]
        loss = loss.pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        opened = loss.item() if opened is None else opened

    after = selection_rate(held_out)
    chance = 1.0 / choices
    assert after > before + 0.15, (
        "top-k selection of the pointed-at block did not improve on held-out batches: "
        "%.3f -> %.3f (chance %.3f)" % (before, after, chance))
    assert after > 2.0 * chance, (
        "selection stayed near chance: %.3f against %.3f" % (after, chance))


@pytest.mark.parametrize("window", [1, 8, BLOCK, BLOCK + 1, 3 * BLOCK])
def test_local_window_is_always_routed(window):
    """Every query keeps ``local_window`` tokens of history whatever the router scores.

    The diagonal block alone does not give this: a query at offset 0 of a block needs the
    block in front of it, and top-k is free to drop that.
    """
    torch.manual_seed(0)
    layer = bare_layer(csa2_local_window=window)
    hidden = torch.randn(2, 6 * BLOCK, layer.config.hidden_size)
    allowed, _ = layer.route(hidden, layer.index_k_proj(hidden))

    blocks = allowed.shape[-1]
    for query_block in range(blocks):
        for position in (0, BLOCK // 2, BLOCK - 1):
            earliest = max(0, query_block * BLOCK + position - window)
            for key_block in range(earliest // BLOCK, query_block + 1):
                assert bool(allowed[:, query_block, key_block].all()), (
                    "window %d: query block %d lost key block %d"
                    % (window, query_block, key_block))


@pytest.mark.parametrize("scale", [1.0, 8.0])
@pytest.mark.parametrize("window", [0, 8])
def test_routing_does_not_depend_on_a_token_s_own_future(window, scale):
    """Mutating a block's last token must not change what its earlier tokens may read.

    One routing decision serves a whole block, so pooling the block's queries to make it
    -- an amax over all of them -- lets a token's history depend on tokens that follow
    it. The causal mask cannot undo that: it constrains what is read, not what chose it.

    The mutation has to land *inside* a block. A cut at a block boundary, which is what
    the original prefix-invariance check did, cannot see this at all. Before the fix this
    fired on 18 of 200 mutations at ``scale`` 1.0 and 191 of 200 at 8.0.
    """
    torch.manual_seed(0)
    blocks = 10
    layer = bare_layer(csa2_local_window=window, csa2_top_k=2 * BLOCK)
    width = layer.config.hidden_size
    block = blocks - 1
    victim = block * BLOCK + BLOCK - 1

    for _ in range(25):
        hidden = torch.randn(1, blocks * BLOCK, width)
        before, _ = layer.route(hidden, layer.index_k_proj(hidden))
        changed = hidden.clone()
        changed[:, victim] += scale * torch.randn(width)
        after, _ = layer.route(changed, layer.index_k_proj(changed))
        assert torch.equal(before[0, block], after[0, block]), (
            "block %d read %s before the mutation and %s after, but every token in it "
            "below offset %d has an unchanged prefix"
            % (block, before[0, block].nonzero().flatten().tolist(),
               after[0, block].nonzero().flatten().tolist(), BLOCK - 1))


def test_block_mask_refuses_to_read_the_future():
    """A future block passed as `full` would be attended with no mask evaluated at all."""
    layer = bare_layer()
    blocks = 4
    allowed = torch.ones(1, blocks, blocks, dtype=torch.bool)
    # `to_dense` is block-level, so the diagonal is present either way; what must not
    # survive is any block strictly above it.
    dense = layer.block_mask(allowed).to_dense().bool()[0, 0]
    rows = torch.arange(blocks)
    assert not bool(dense[rows.view(-1, 1) < rows.view(1, -1)].any())
    assert bool(dense[rows.view(-1, 1) >= rows.view(1, -1)].all())


@cuda
def test_routing_report_describes_the_last_forward():
    """Density alone cannot spot a collapsed router, and the entropy has to be honest.

    Two traps this covers. A router that has fallen back onto the local window still
    reports a healthy density, because the window is open regardless of score -- so
    `selected` measures only the blocks top-k was free to choose. And entropy normalized
    by the blocks actually used would score an even split over two blocks as a perfect
    1.0, which is the collapse it exists to catch, so it is normalized by the blocks
    available instead.
    """
    from distillkit.models.qwen35.csa2 import routing_report

    torch.manual_seed(0)
    model = Qwen35WidenedForCausalLM(csa2_config(csa2_top_k=2 * BLOCK)).cuda()
    model.eval()
    assert routing_report(model) == [], "nothing has run yet"

    tokens = torch.randint(0, 64, (4, 8 * BLOCK), device="cuda")
    with torch.no_grad():
        model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                    use_cache=False)
    rows = routing_report(model)
    assert [r["mode"] for r in rows] == MODES
    for row in rows:
        assert row["blocks"] == 8
        assert 0.0 < row["density"] < 1.0, row
        assert 0.0 < row["selected"] < 1.0, row
        assert 0.0 < row["entropy"] <= 1.0, row

    # A router pinned to one key block must read as collapsed, not as healthy density.
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


@pytest.mark.parametrize("mode,pieces", [("full", 4), ("reindex", 2), ("reuse", 1)])
def test_fused_projection_matches_separate_ones(mode, pieces):
    """One multiply over the stream must give what four multiplies over it gave.

    A Full layer projects the query, the latent, the index keys and the index queries
    from the same `[batch, tokens, hidden]` tensor, and three of the four are narrow
    enough that reading that tensor dominates. Sharing the multiply is only worth doing
    if the split puts every piece back exactly where it was -- a wrong order or width
    would train perfectly well and mean something else.
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
        expected += [layer.kv_a_proj, layer.index_k_proj]
    if mode in ("full", "reindex"):
        expected.append(layer.index_q_proj)
    for part, module in zip(projected, expected):
        assert torch.allclose(part, module(hidden), atol=1e-6, rtol=1e-6)
        assert part.shape[-1] == module.out_features


def test_routing_stays_causal_and_sparse():
    torch.manual_seed(0)
    layer = bare_layer(csa2_local_window=0, csa2_top_k=BLOCK)
    hidden = torch.randn(2, 8 * BLOCK, layer.config.hidden_size)
    allowed, _ = layer.route(hidden, layer.index_k_proj(hidden))

    blocks = allowed.shape[-1]
    rows = torch.arange(blocks)
    causal = rows.view(-1, 1) >= rows.view(1, -1)
    assert not bool((allowed & ~causal.unsqueeze(0)).any())
    # Something has to be dropped, or the sparsity is decorative.
    assert allowed.sum() < causal.sum() * allowed.shape[0]


def test_router_columns_carry_a_bounded_cosine():
    """The appended columns must reproduce ``index_gate * cos`` once the kernel scales.

    This is the whole gradient path, and it is invisible in any output: get the scaling
    wrong and the router still trains, just against a logit shift of the wrong size.
    """
    torch.manual_seed(0)
    layer = bare_layer()
    with torch.no_grad():
        layer.index_gate.fill_(0.7)
    hidden = torch.randn(2, 2 * BLOCK, layer.config.hidden_size)
    keys = layer.index_k_proj(hidden)
    _, effective = layer.route(hidden, keys)

    query_extra, key_extra = layer.router_columns(effective, keys)
    contributed = (query_extra @ key_extra.transpose(-1, -2)) * layer.scaling
    cosine = torch.nn.functional.cosine_similarity(
        effective.unsqueeze(2), keys.unsqueeze(1), dim=-1)
    assert torch.allclose(contributed.squeeze(1), 0.7 * cosine, atol=1e-5)
    assert contributed.abs().max() <= 0.7 + 1e-5


def test_the_tile_follows_bytes_rather_than_the_head_alone():
    """float32 doubles every tile, so the same head that fits in bfloat16 may not.

    The 2B's dense width asked 149568 of 101376 in float32 while the bfloat16 tile for the
    same shape fit, and nothing in the rule saw it: shared memory holds bytes and the rule
    was reading elements. The indexer's teacher runs float32, and a self-distillation runs
    the student's own attention as that teacher, so those layers reach this path.
    """
    layer = bare_layer()
    assert layer._kernel_options(256, 2)["fwd_num_stages"] == 3
    # Same head, twice the bytes: it has to land where the wider bfloat16 head lands.
    assert layer._kernel_options(256, 4) == layer._kernel_options(320, 2)
    # And the widest case steps down again rather than reusing a tile that cannot fit.
    assert layer._kernel_options(320, 4)["fwd_BLOCK_N"] == 16
    for element_size in (2, 4):
        for width in (256, 320):
            options = layer._kernel_options(width, element_size)
            assert layer.block_size % options["fwd_BLOCK_M"] == 0
            assert layer.block_size % options["fwd_BLOCK_N"] == 0


def test_both_halves_of_the_kernel_are_named_with_their_prefix():
    """A bare key reaches both lowerings, so the forward would set the backward's stages.

    Inductor strips `fwd_` in the forward and drops `bwd_`, and the reverse in the
    backward -- but it leaves an unprefixed key alone in each, and its `setdefault` then
    finds one already there. The backward's own table is what has to be overridden here:
    on SM86 it hands a head of exactly 256 a 64x64x64x64 tile that asks 168960 of 101376,
    while a wider one gets 16x16x16x16 and fits. A reuse layer carries no router columns,
    so it is the narrow one, and it is the only layer that cannot compile its backward.
    """
    layer = bare_layer()
    for width in (256, 320):
        options = layer._kernel_options(width, 2)
        assert all(k.startswith(("fwd_", "bwd_")) for k in options), options
        assert options["bwd_BLOCK_M1"] == options["bwd_BLOCK_N1"]
        assert layer.block_size % options["bwd_BLOCK_M1"] == 0
    # The narrow head is the one Inductor over-sizes, so it must not be left alone.
    assert layer._kernel_options(256, 2)["bwd_BLOCK_M1"] == 32
    assert layer._kernel_options(320, 2)["bwd_BLOCK_M1"] == 16


@pytest.mark.parametrize("width,stages", [(256, 3), (320, 1)])
def test_kernel_tiles_are_chosen_rather_than_left_to_inductor(width, stages):
    """Inductor will not pick a tile that fits, so the tile has to be picked here.

    It rounds the key/query head to a power of two, then reads its tile from a table keyed
    on the *unrounded* width without comparing the result to the device, and when the
    result does not fit it drops the config instead of shrinking it -- the choice list
    empties and the compile ends in "No valid triton configs". Measured against SM86's
    101376 bytes at the real student's two widths, its defaults ask 151552 at 256 and
    167936 at 320. The 256 case is the dense path, carrying no router columns at all.
    """
    layer = bare_layer()
    options = layer._kernel_options(width)
    assert options["fwd_num_stages"] == stages
    # Inductor refuses outright when these do not divide the mask's block size.
    assert layer.block_size % options["fwd_BLOCK_M"] == 0
    assert layer.block_size % options["fwd_BLOCK_N"] == 0


def test_the_router_columns_are_what_widens_the_kernel_tile():
    """Sixteen columns of index buy a whole extra power of two, and that is the blocker.

    At the real student's head of 256 the router's columns are not a rounding error on the
    tile, they double it: every query and key buffer in shared memory is sized to 512 to
    carry them. It is the reason the sparse path needs a shallower pipeline than the dense
    one over the same head.
    """
    layer = bare_layer(csa2_index_dim=64)
    head = 256
    assert 1 << (head - 1).bit_length() == head
    assert 1 << (head + layer.index_dim - 1).bit_length() == 2 * head
    assert (layer._kernel_options(head)["fwd_num_stages"]
            > layer._kernel_options(head + layer.index_dim)["fwd_num_stages"])


def test_block_mask_matches_a_scanned_mask():
    """`from_kv_blocks` built by hand against `create_block_mask` evaluating the rule."""
    from torch.nn.attention.flex_attention import create_block_mask

    torch.manual_seed(0)
    layer = bare_layer()
    blocks, batch = 6, 2
    rows = torch.arange(blocks)
    causal = rows.view(-1, 1) >= rows.view(1, -1)
    allowed = (torch.rand(batch, blocks, blocks) < 0.5) & causal.unsqueeze(0)
    allowed |= torch.eye(blocks, dtype=torch.bool).unsqueeze(0)

    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & allowed[b, q_idx // BLOCK, kv_idx // BLOCK]

    scanned = create_block_mask(mask_mod, B=batch, H=None, Q_LEN=blocks * BLOCK,
                                KV_LEN=blocks * BLOCK, device="cpu", BLOCK_SIZE=BLOCK)
    built = layer.block_mask(allowed)
    assert torch.equal(built.to_dense(), scanned.to_dense())


@cuda
def test_sparse_kernel_matches_a_dense_reference():
    """Output and gradients against SDPA over the same mask and the same bias.

    The block mask, the full/partial split and the score bias are all only as good as
    what the kernel does with them, and none of those are visible in a loss curve.
    """
    torch.manual_seed(0)
    from torch.nn.attention.flex_attention import flex_attention

    layer = bare_layer().cuda()
    batch, blocks = 2, 4
    seq = blocks * BLOCK
    shape = (batch, HEADS, seq, HEAD_DIM)
    query, key, value = (torch.randn(shape, device="cuda", dtype=torch.float32,
                                     requires_grad=True) for _ in range(3))

    rows = torch.arange(blocks, device="cuda")
    causal = rows.view(-1, 1) >= rows.view(1, -1)
    allowed = ((torch.rand(batch, blocks, blocks, device="cuda") < 0.5)
               & causal.unsqueeze(0)) | torch.eye(blocks, dtype=torch.bool,
                                                  device="cuda").unsqueeze(0)
    sparse = torch.compile(flex_attention, dynamic=False)(
        query, key, value, block_mask=layer.block_mask(allowed), scale=HEAD_DIM ** -0.5)
    sparse.pow(2).sum().backward()
    got = [t.grad.clone() for t in (query, key, value)]
    for tensor in (query, key, value):
        tensor.grad = None

    positions = torch.arange(seq, device="cuda")
    token_mask = ((positions.view(-1, 1) >= positions.view(1, -1)).unsqueeze(0)
                  & allowed.repeat_interleave(BLOCK, 1).repeat_interleave(BLOCK, 2))
    scores = query @ key.transpose(-1, -2) * HEAD_DIM ** -0.5
    scores = scores.masked_fill(~token_mask.unsqueeze(1), float("-inf"))
    dense = scores.softmax(-1) @ value
    dense.pow(2).sum().backward()

    assert torch.allclose(sparse, dense, atol=2e-4, rtol=2e-4)
    for tensor, reference in zip((query, key, value), got):
        assert torch.allclose(reference, tensor.grad, atol=2e-3, rtol=2e-3)


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
def test_shapes_the_block_kernel_cannot_take_route_per_token(length):
    """A ragged length and a decode step are ordinary cases, not refusals.

    They used to raise, because a ``BlockMask`` groups queries as well as keys: a
    one-token step computes zero query blocks and a ragged one loses its tail. The
    reference groups only the key axis and takes its top-k per query token, so both are
    shapes it simply routes. `_blocked` picks the path, and the kernel keeps the whole
    query blocks it needs.
    """
    layer = bare_layer()
    layer.bus = SparseIndexBus()
    hidden = torch.randn(1, length, layer.config.hidden_size)
    width = int(layer.head_dim * 0.25)
    position = (torch.ones(1, length, width), torch.zeros(1, length, width))
    assert layer._blocked(length, None) is (length == BLOCK)
    if length == BLOCK:
        # The blocked path is what already had coverage; running it here would only be
        # asking CPU Inductor to build a Triton kernel, which needs a compiler this
        # environment does not have. Which path the shape takes is the claim.
        return
    with torch.no_grad():
        out, _ = layer.forward(hidden, position_embeddings=position)
    assert out.shape == hidden.shape
    assert layer.last_token_routed
    # Selected positions, not blocks: the granularity is the whole point of the path.
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
    # The latent and the index keys share the key slot; the rotary slice is the value.
    keys = out.past_key_values.layers[full[0]].keys
    assert keys.shape[-1] == owner.latent + owner.index_dim
    assert owner.cached_numbers_per_token() == owner.latent + owner.rope_dim + owner.index_dim


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


