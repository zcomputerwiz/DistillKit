"""Unit gates for the GGUF->HF student converter (no real GGUF file needed).

Covers the name mapping, config derivation from qwen35.* KV values, expected
key-set geometry, and the BF16/F32 decode path against torch's native bf16.
The full-file conversion itself is exercised by scratch/verify_converted_student.py.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

# gguf ships only in the optional "ngram" extra; skip collection without it so a
# base/dev install can still run the rest of the suite (matches test_ngram_table).
gguf = pytest.importorskip("gguf", reason="gguf package not installed")
GGMLQuantizationType = gguf.GGMLQuantizationType

from distillkit.convert_gguf_student import (
    EOS_TOKEN_ID,
    _Dims,
    _decode_tensor,
    _invert_ssm_a,
    _unreorder_v_heads,
    convert_gguf_to_hf,
    derive_text_config,
    expected_key_set,
    map_gguf_tensor,
)


def make_kv(**overrides):
    """The real Qwen3.8-4B-BF16.gguf qwen35.* metadata (plus general.architecture)."""
    kv = {
        "general.architecture": "qwen35",
        "qwen35.block_count": 33,
        "qwen35.nextn_predict_layers": 1,
        "qwen35.embedding_length": 2560,
        "qwen35.feed_forward_length": 9216,
        "qwen35.context_length": 262144,
        "qwen35.full_attention_interval": 4,
        "qwen35.attention.head_count": 16,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256,
        "qwen35.attention.value_length": 256,
        "qwen35.attention.layer_norm_rms_epsilon": 9.999999974752427e-07,
        "qwen35.rope.dimension_count": 64,
        "qwen35.rope.dimension_sections": [11, 11, 10, 0],
        "qwen35.rope.freq_base": 10000000.0,
        "qwen35.ssm.conv_kernel": 4,
        "qwen35.ssm.state_size": 128,
        "qwen35.ssm.group_count": 16,
        "qwen35.ssm.time_step_rank": 32,
        "qwen35.ssm.inner_size": 4096,
    }
    kv.update(overrides)
    return kv


def test_derive_text_config_matches_stock_geometry():
    cfg = derive_text_config(make_kv(), vocab_size=248320)
    assert cfg["model_type"] == "qwen3_5_text"
    assert cfg["architectures"] == ["Qwen3_5ForCausalLM"]
    assert cfg["num_hidden_layers"] == 32
    assert cfg["hidden_size"] == 2560
    assert cfg["intermediate_size"] == 9216
    assert cfg["head_dim"] == 256
    assert cfg["num_attention_heads"] == 16
    assert cfg["num_key_value_heads"] == 4
    assert cfg["max_position_embeddings"] == 262144
    assert cfg["rms_norm_eps"] == pytest.approx(1e-6, rel=1e-5)
    assert cfg["tie_word_embeddings"] is True
    assert cfg["vocab_size"] == 248320
    assert cfg["eos_token_id"] == EOS_TOKEN_ID == 248044
    # mamba geometry
    assert cfg["linear_conv_kernel_dim"] == 4
    assert cfg["linear_key_head_dim"] == 128
    assert cfg["linear_value_head_dim"] == 128
    assert cfg["linear_num_key_heads"] == 16
    assert cfg["linear_num_value_heads"] == 32
    # layer pattern: full attention every 4th layer, 0-indexed at 3
    assert cfg["layer_types"] == [
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(32)
    ]
    assert cfg["layer_types"].count("full_attention") == 8
    # rope
    rope = cfg["rope_parameters"]
    assert rope["mrope_section"] == [11, 11, 10]  # trailing zero stripped
    assert rope["rope_theta"] == 10000000.0
    assert rope["partial_rotary_factor"] == pytest.approx(0.25)
    assert rope["mrope_interleaved"] is True


def test_derive_text_config_rejects_bad_metadata():
    kv = make_kv()
    del kv["qwen35.block_count"]
    with pytest.raises(KeyError, match="block_count"):
        derive_text_config(kv, vocab_size=1)
    with pytest.raises(ValueError, match="value_length"):
        derive_text_config(make_kv(**{"qwen35.attention.value_length": 128}), vocab_size=1)
    with pytest.raises(ValueError, match="multiple of ssm.state_size"):
        derive_text_config(make_kv(**{"qwen35.ssm.inner_size": 4000}), vocab_size=1)
    with pytest.raises(ValueError, match="mrope section sum"):
        derive_text_config(make_kv(**{"qwen35.rope.dimension_sections": [11, 11, 9, 0]}), vocab_size=1)
    with pytest.raises(ValueError, match="divide head_dim"):
        # sections still tile (64/2), but 64 does not divide head_dim 200
        derive_text_config(
            make_kv(**{"qwen35.attention.key_length": 200, "qwen35.attention.value_length": 200}),
            vocab_size=1,
        )


def test_map_gguf_tensor_top_level_and_mtp():
    assert map_gguf_tensor("token_embd.weight", 32) == ("model.embed_tokens.weight", "copy", torch.bfloat16)
    assert map_gguf_tensor("output_norm.weight", 32) == ("model.norm.weight", "copy", torch.bfloat16)
    # trailing MTP block is skipped, not mapped
    assert map_gguf_tensor("blk.32.attn_qkv.weight", 32) is None
    assert map_gguf_tensor("blk.32.nextn.eh_proj.weight", 32) is None
    with pytest.raises(KeyError):
        map_gguf_tensor("foo.bar", 32)


def test_map_gguf_tensor_layer_suffixes():
    linear = {
        "attn_norm.weight": ("input_layernorm.weight", torch.bfloat16),
        "post_attention_norm.weight": ("post_attention_layernorm.weight", torch.bfloat16),
        "ffn_gate.weight": ("mlp.gate_proj.weight", torch.bfloat16),
        "ffn_up.weight": ("mlp.up_proj.weight", torch.bfloat16),
        "ffn_down.weight": ("mlp.down_proj.weight", torch.bfloat16),
        "attn_qkv.weight": ("linear_attn.in_proj_qkv.weight", torch.bfloat16),
        "attn_gate.weight": ("linear_attn.in_proj_z.weight", torch.bfloat16),
        "ssm_alpha.weight": ("linear_attn.in_proj_a.weight", torch.bfloat16),
        "ssm_beta.weight": ("linear_attn.in_proj_b.weight", torch.bfloat16),
        "ssm_conv1d.weight": ("linear_attn.conv1d.weight", torch.bfloat16),
        "ssm_out.weight": ("linear_attn.out_proj.weight", torch.bfloat16),
        "ssm_norm.weight": ("linear_attn.norm.weight", torch.float32),
        "ssm_a": ("linear_attn.A_log", torch.float32),
        "ssm_dt.bias": ("linear_attn.dt_bias", torch.bfloat16),
    }
    for suffix, (key, dtype) in linear.items():
        hf_key, transform, got_dtype = map_gguf_tensor(f"blk.0.{suffix}", 32)
        assert hf_key == f"model.layers.0.{key}", suffix
        assert got_dtype is dtype, suffix
        assert transform in ("copy", "conv"), suffix

    full = {
        "attn_q.weight": ("self_attn.q_proj.weight",),
        "attn_k.weight": ("self_attn.k_proj.weight",),
        "attn_v.weight": ("self_attn.v_proj.weight",),
        "attn_output.weight": ("self_attn.o_proj.weight",),
        "attn_q_norm.weight": ("self_attn.q_norm.weight",),
        "attn_k_norm.weight": ("self_attn.k_norm.weight",),
    }
    for suffix, (key,) in full.items():
        hf_key, _, _ = map_gguf_tensor(f"blk.3.{suffix}", 32)
        assert hf_key == f"model.layers.3.{key}", suffix

    with pytest.raises(KeyError):
        map_gguf_tensor("blk.0.unknown_suffix.weight", 32)


def test_expected_key_set_geometry():
    cfg = derive_text_config(make_kv(), vocab_size=248320)
    keys = expected_key_set(cfg)
    # 2 top-level + 24 linear layers x 14 + 8 full layers x 11 == stock language_model count
    assert len(keys) == 2 + 24 * 14 + 8 * 11 == 426
    assert "model.layers.0.linear_attn.A_log" in keys
    assert "model.layers.3.self_attn.q_proj.weight" in keys
    assert not any(k.startswith("mtp.") for k in keys)


def test_dims_expected_shapes_and_mismatch():
    cfg = derive_text_config(make_kv(), vocab_size=248320)
    dims = _Dims(cfg, time_step_rank=32)
    H, I = 2560, 9216
    assert dims.expected("linear_attention", "attn_qkv.weight") == (8192, H)
    assert dims.expected("linear_attention", "attn_gate.weight") == (4096, H)
    assert dims.expected("linear_attention", "ssm_conv1d.weight") == (8192, 4)
    assert dims.expected("linear_attention", "ssm_out.weight") == (H, 4096)
    # attn_output_gate doubles the q output
    assert dims.expected("full_attention", "attn_q.weight") == (2 * 16 * 256, H)
    assert dims.expected("full_attention", "attn_k.weight") == (4 * 256, H)
    assert dims.expected("full_attention", "attn_output.weight") == (H, 16 * 256)
    with pytest.raises(ValueError, match="not valid for layer type"):
        dims.expected("full_attention", "ssm_a")
    with pytest.raises(ValueError, match="not valid for layer type"):
        dims.expected("linear_attention", "attn_q.weight")


def _fake_reader_tensor(name, tensor_type, data, n_bytes=None):
    return SimpleNamespace(
        name=name,
        tensor_type=tensor_type,
        data=data,
        n_bytes=n_bytes if n_bytes is not None else int(np.asarray(data).nbytes),
    )


def test_decode_bf16_2d_matches_torch_native():
    reader = SimpleNamespace(byte_order="I")
    torch.manual_seed(0)
    vals = (torch.randn(4, 8) * 3).to(torch.bfloat16)
    raw = np.ascontiguousarray(vals.view(torch.uint8).numpy())  # (out, in*2)
    out = _decode_tensor(reader, _fake_reader_tensor("t", GGMLQuantizationType.BF16, raw))
    assert out.dtype == np.float32
    assert out.shape == (4, 8)
    assert torch.equal(torch.from_numpy(out.copy()), vals.float())


def test_decode_bf16_1d():
    reader = SimpleNamespace(byte_order="I")
    vals = torch.linspace(-2, 2, 16).to(torch.bfloat16)
    raw = np.ascontiguousarray(vals.view(torch.uint8).numpy())  # (n*2,)
    out = _decode_tensor(reader, _fake_reader_tensor("t", GGMLQuantizationType.BF16, raw))
    assert out.shape == (16,)
    assert torch.equal(torch.from_numpy(out.copy()), vals.float())


def test_decode_f32_returns_writable_copy():
    reader = SimpleNamespace(byte_order="I")
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = _decode_tensor(reader, _fake_reader_tensor("t", GGMLQuantizationType.F32, src))
    assert out.shape == (3, 4)
    assert np.array_equal(out, src)
    out[0, 0] = -1.0  # must not touch the source memmap
    assert src[0, 0] != -1.0


def test_decode_rejects_big_endian_and_unknown_type():
    reader_be = SimpleNamespace(byte_order="S")
    with pytest.raises(ValueError, match="little-endian"):
        _decode_tensor(reader_be, _fake_reader_tensor("t", GGMLQuantizationType.F32, np.zeros(4, np.float32)))
    reader = SimpleNamespace(byte_order="I")
    with pytest.raises(ValueError, match="Unsupported GGML type"):
        _decode_tensor(reader, _fake_reader_tensor("t", GGMLQuantizationType.Q8_0, np.zeros(16, np.uint8)))


def _forward_reorder_v_heads(x, dim, num_k_heads, num_v_per_k, head_dim):
    """Independent copy of llama.cpp's HF->GGUF grouped->tiled reorder."""
    shape = list(x.shape)
    if dim < 0:
        dim += len(shape)
    t = x.reshape(*shape[:dim], num_k_heads, num_v_per_k, head_dim, *shape[dim + 1:])
    axes = list(range(t.ndim))
    axes[dim], axes[dim + 1] = axes[dim + 1], axes[dim]
    return t.transpose(axes).reshape(shape)


@pytest.mark.parametrize("k,v,d", [(16, 2, 128), (16, 2, 1), (2, 3, 1), (4, 5, 7)])
def test_unreorder_v_heads_inverts_forward_rows(k, v, d):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((k * v * d,))
    y = _forward_reorder_v_heads(x, 0, k, v, d)
    assert np.array_equal(_unreorder_v_heads(y, 0, k, v, d), x)


def test_unreorder_v_heads_inverts_forward_columns():
    # out_proj case: reorder along dim=1 (input dimension)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((64, 16 * 2 * 8))
    y = _forward_reorder_v_heads(x, 1, 16, 2, 8)
    assert np.array_equal(_unreorder_v_heads(y, 1, 16, 2, 8), x)


def test_invert_ssm_a_round_trip_and_guard():
    rng = np.random.default_rng(2)
    a_log = rng.standard_normal((32,)) - 2.0  # plausible log-decay values
    stored = -np.exp(a_log)
    assert np.allclose(_invert_ssm_a(stored), a_log, rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError, match="strictly negative"):
        _invert_ssm_a(np.array([0.5, -0.5]))


def _write_synthetic_qwen35_gguf(path, *, inner_size, a_log):
    """Write a minimal 2-block qwen35 GGUF with controlled ``ssm_a`` values.

    Both blocks are linear_attention (full_attention_interval=4). Header dims
    use llama.cpp ``[in, out]`` order while the byte stream is the HF
    ``[out, in]`` array, mirroring real files.
    """
    from gguf import GGMLQuantizationType, GGUFValueType, GGUFWriter

    H, I, vocab = 32, 64, 64
    key_heads, state_size = 2, 8
    value_heads = inner_size // state_size
    tsr = value_heads  # A_log/dt_bias are per value head
    rng = np.random.default_rng(0)

    def bf16_bytes(array):
        return (
            torch.from_numpy(np.ascontiguousarray(array))
            .to(torch.bfloat16)
            .view(torch.uint8)
            .numpy()
        )

    writer = GGUFWriter(str(path), "qwen35")
    for key, value in {
        "block_count": 2, "nextn_predict_layers": 0, "embedding_length": H,
        "feed_forward_length": I, "context_length": 256, "full_attention_interval": 4,
        "attention.head_count": 2, "attention.head_count_kv": 1,
        "attention.key_length": 16, "attention.value_length": 16,
        "rope.dimension_count": 8,
        "ssm.conv_kernel": 4, "ssm.state_size": state_size,
        "ssm.group_count": key_heads, "ssm.time_step_rank": tsr,
        "ssm.inner_size": inner_size,
    }.items():
        writer.add_key_value(f"qwen35.{key}", value, GGUFValueType.INT32)
    for key, value in {
        "attention.layer_norm_rms_epsilon": 1e-6, "rope.freq_base": 10000.0,
    }.items():
        writer.add_key_value(f"qwen35.{key}", value, GGUFValueType.FLOAT32)
    writer.add_key_value(
        "qwen35.rope.dimension_sections", [4, 0], GGUFValueType.ARRAY, sub_type=GGUFValueType.INT32
    )

    # The writer stores the passed array's own shape (halving the last axis for
    # BF16) and reverses dims when emitting the file, so passing the HF
    # [out, in] byte array with no raw_shape yields llama.cpp [in, out] header
    # dims and an aligned data view.
    def add_linear(name, hf_array):
        writer.add_tensor(name, bf16_bytes(hf_array), raw_dtype=GGMLQuantizationType.BF16)

    def add_vector(name, hf_array, f32=False):
        if f32:
            writer.add_tensor(name, np.ascontiguousarray(hf_array, dtype=np.float32))
        else:
            writer.add_tensor(name, bf16_bytes(hf_array), raw_dtype=GGMLQuantizationType.BF16)

    key_dim, val_dim = key_heads * state_size, value_heads * state_size
    qkv_dim = 2 * key_dim + val_dim

    emb = rng.standard_normal((vocab, H))
    writer.add_tensor("token_embd.weight", bf16_bytes(emb), raw_dtype=GGMLQuantizationType.BF16)
    add_vector("output_norm.weight", rng.standard_normal(H) + 1.0)

    stored_a_log = -np.exp(np.asarray(a_log, dtype=np.float32))
    for layer in range(2):
        prefix = f"blk.{layer}."
        add_vector(prefix + "attn_norm.weight", rng.standard_normal(H) + 1.0)
        add_vector(prefix + "post_attention_norm.weight", rng.standard_normal(H) + 1.0)
        add_linear(prefix + "ffn_gate.weight", rng.standard_normal((I, H)))
        add_linear(prefix + "ffn_up.weight", rng.standard_normal((I, H)))
        add_linear(prefix + "ffn_down.weight", rng.standard_normal((H, I)))
        add_linear(prefix + "attn_qkv.weight", rng.standard_normal((qkv_dim, H)))
        add_linear(prefix + "attn_gate.weight", rng.standard_normal((val_dim, H)))
        add_linear(prefix + "ssm_alpha.weight", rng.standard_normal((tsr, H)))
        add_linear(prefix + "ssm_beta.weight", rng.standard_normal((tsr, H)))
        add_linear(prefix + "ssm_conv1d.weight", rng.standard_normal((qkv_dim, 4)))
        add_linear(prefix + "ssm_out.weight", rng.standard_normal((H, val_dim)))
        add_vector(prefix + "ssm_norm.weight", rng.standard_normal(state_size), f32=True)
        # Unequal-head files store ssm_a in llama.cpp's tiled V-head order;
        # equal-head files store it as-is. Both store -exp(A_log).
        stored = stored_a_log
        if value_heads != key_heads:
            stored = _forward_reorder_v_heads(stored, 0, key_heads, value_heads // key_heads, 1)
        add_vector(prefix + "ssm_a", stored, f32=True)
        add_vector(prefix + "ssm_dt.bias", rng.standard_normal(tsr))
    # The 0.19 writer only emits bytes through this explicit sequence.
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.mark.parametrize(
    ("inner_size", "a_log"),
    [
        pytest.param(16, [0.5, -1.2], id="equal-heads"),  # 2 value heads == 2 key heads
        pytest.param(32, [0.5, -1.2, 2.0, -0.7], id="unequal-heads"),  # 4 vs 2
    ],
)
def test_synthetic_gguf_round_trip_recovers_a_log(tmp_path, inner_size, a_log):
    gguf_path = tmp_path / "synthetic.gguf"
    _write_synthetic_qwen35_gguf(gguf_path, inner_size=inner_size, a_log=a_log)
    out_dir = tmp_path / "out"
    summary = convert_gguf_to_hf(gguf_path, out_dir)
    assert summary["tensors"] == 2 + 2 * 14  # top-level + two linear layers

    from safetensors.torch import load_file

    weights = load_file(str(out_dir / "model.safetensors"))
    expected = np.asarray(a_log, dtype=np.float32)
    for layer in range(2):
        a_log_out = weights[f"model.layers.{layer}.linear_attn.A_log"].numpy()
        assert a_log_out.dtype == np.float32
        # Distinct values make any skipped/misapplied head reorder detectable.
        np.testing.assert_allclose(a_log_out, expected, rtol=1e-6, atol=1e-7)
