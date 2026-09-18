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
from distillkit.models.qwen35.csa2 import Qwen35SparseLatentAttention

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
                   ("index_q_proj", "index_k_proj", "index_weight", "index_gate")):
            continue
        checked += 1
        assert parameter.grad is not None, "%s took no gradient" % name
        assert torch.isfinite(parameter.grad).all(), "%s gradient is not finite" % name
        assert parameter.grad.abs().sum() > 0, "%s gradient is all zero" % name
    # Two full layers own queries, keys, weights and a gate; the reindex layer owns
    # everything but the keys.
    assert checked == 11


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


def test_ragged_length_and_cache_are_refused():
    """Both mis-route in silence: a short tail is dropped, a decode step routes nothing."""
    layer = bare_layer()
    layer.bus = object()
    hidden = torch.randn(1, BLOCK + 1, layer.config.hidden_size)
    with pytest.raises(ValueError, match="multiple of csa2_block_size"):
        layer.forward(hidden, position_embeddings=None)
    with pytest.raises(ValueError, match="multiple of csa2_block_size"):
        layer.forward(torch.randn(1, 1, layer.config.hidden_size),
                      position_embeddings=None)
    with pytest.raises(RuntimeError, match="training-only"):
        layer.forward(torch.randn(1, BLOCK, layer.config.hidden_size),
                      position_embeddings=None, past_key_values=object())


def test_padding_and_gradient_checkpointing_are_refused():
    """Padding is not representable per block; checkpointing replays the bus out of order."""
    model = Qwen35WidenedForCausalLM(csa2_config())
    tokens = torch.randint(1, 64, (2, BLOCK))
    mask = torch.ones_like(tokens)
    mask[0, -3:] = 0
    with pytest.raises(ValueError, match="cannot represent padding"):
        model.model(input_ids=tokens, attention_mask=mask, use_cache=False)

    model.model.gradient_checkpointing = True
    model.train()
    with pytest.raises(RuntimeError, match="gradient checkpointing"):
        model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                    use_cache=False)


def test_mla_caches_the_latent_not_the_expanded_heads():
    """The cache holds ``latent + rope_dim``, and replaying it reproduces the dense run.

    Handing `update` the assembled per-head keys caches ``2 * num_heads * head_dim``
    instead -- four times the GQA this replaces, while `cached_numbers_per_token` keeps
    reporting the latent width. Tested on the attention module rather than the stack:
    the widened model's chunked-cache continuation is separately broken, with or without
    MLA, so a whole-model comparison would fail for an unrelated reason.
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
    # The GQA it replaces caches a key and a value per key/value head.
    assert width < 2 * config.num_attention_heads * HEAD_DIM
