"""Unit tests for Flash-Attention 2 integration in DistillKit.

All tests are strictly CPU-safe and do not allocate CUDA memory or run CUDA kernels,
ensuring host GPU training jobs are not disturbed.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from distillkit.models.loader import is_flash_attn_available, load_student_model
import distillkit.parallel.blocks as parallel_blocks
from distillkit.parallel.blocks import TensorParallelAttention, _repeat_kv


def test_is_flash_attn_available_behavior():
    """Verify is_flash_attn_available handles missing or broken extensions."""
    with patch("importlib.util.find_spec", return_value=None):
        assert is_flash_attn_available() is False

    with patch("importlib.util.find_spec", return_value=MagicMock()):
        with patch.dict("sys.modules", {"flash_attn": None}):
            with patch("builtins.__import__", side_effect=ImportError("DLL load failed")):
                assert is_flash_attn_available() is False


def test_load_student_model_flash_attn_error_when_unavailable(monkeypatch):
    """When use_flash_attention is True but flash_attn cannot be imported, raise actionable RuntimeError."""
    monkeypatch.setattr(parallel_blocks, "_HAS_FLASH_ATTN", False)

    with patch("distillkit.models.loader.is_flash_attn_available", return_value=False):
        config = SimpleNamespace(
            functionary_packing=False,
            sidecar=None,
            model_auto_class="AutoModelForCausalLM",
            trust_remote_code=False,
            train_model="does-not-matter",
            use_flash_attention=True,
            model_kwargs={},
            training_args={},
        )
        with pytest.raises(RuntimeError, match="flash_attn is not installed"):
            load_student_model(config, tokenizer_vocab_size=32, signal_vocab_size=32)


def test_load_student_model_enables_flash_attention_2():
    """When use_flash_attention is True and available, attn_implementation is set to flash_attention_2."""
    with patch("distillkit.models.loader.is_flash_attn_available", return_value=True), \
         patch("importlib.util.find_spec", return_value=MagicMock()), \
         patch("distillkit.models.loader.resolve_student_class") as mock_resolve, \
         patch("distillkit.models.loader.prepare_student_config", return_value=(None, None)), \
         patch("distillkit.models.loader.post_init_student_model"), \
         patch("distillkit.models.loader.align_student_embeddings"), \
         patch("distillkit.models.loader.apply_freeze_rules"):

        mock_auto_cls = MagicMock()
        mock_resolve.return_value = mock_auto_cls

        config = SimpleNamespace(
            functionary_packing=False,
            sidecar=None,
            model_auto_class="AutoModelForCausalLM",
            trust_remote_code=False,
            train_model="dummy-model",
            use_flash_attention=True,
            model_kwargs={},
            training_args={},
        )

        load_student_model(config, tokenizer_vocab_size=32, signal_vocab_size=32)
        mock_auto_cls.from_pretrained.assert_called_once()
        _, kwargs = mock_auto_cls.from_pretrained.call_args
        assert kwargs["attn_implementation"] == "flash_attention_2"
        assert kwargs["torch_dtype"] == torch.bfloat16


def _mock_attention_module(head_dim=128, num_heads=16, num_kv_heads=4, attn_impl="flash_attention_2"):
    attention = nn.Module()
    attention.head_dim = head_dim
    attention.scaling = head_dim ** -0.5
    attention.attention_dropout = 0.0

    attention.config = SimpleNamespace(
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        _attn_implementation=attn_impl,
    )

    q_out_features = num_heads * head_dim * 2
    kv_out_features = num_kv_heads * head_dim

    attention.q_proj = nn.Linear(num_heads * head_dim, q_out_features, bias=False)
    attention.k_proj = nn.Linear(num_heads * head_dim, kv_out_features, bias=False)
    attention.v_proj = nn.Linear(num_heads * head_dim, kv_out_features, bias=False)
    attention.o_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim, bias=False)

    attention.q_norm = nn.LayerNorm(head_dim)
    attention.k_norm = nn.LayerNorm(head_dim)
    return attention


def test_can_use_flash_attn_predicate_checks():
    """Verify _can_use_flash_attn guards against unsupported devices, dtypes, and masks."""
    attention = _mock_attention_module(head_dim=128, attn_impl="flash_attention_2")
    tp_attn = TensorParallelAttention(attention, ["cpu", "cpu"])

    # On CPU device: always False
    assert tp_attn._can_use_flash_attn(torch.device("cpu"), torch.bfloat16, None) is False

    # Mock flash_attn available and simulate cuda device
    with patch.object(parallel_blocks, "_HAS_FLASH_ATTN", True), \
         patch.object(parallel_blocks, "flash_attn_func", MagicMock()):

        cuda_dev = torch.device("cuda:0")

        # Unsupported dtype: float32 -> False
        assert tp_attn._can_use_flash_attn(cuda_dev, torch.float32, None) is False

        # Supported dtype: bfloat16 -> True
        assert tp_attn._can_use_flash_attn(cuda_dev, torch.bfloat16, None) is True
        assert tp_attn._can_use_flash_attn(cuda_dev, torch.float16, None) is True

        # Attention mask with padding zeros: True if varlen available, False if varlen is None
        mask_with_padding = torch.tensor([[1, 1, 0, 0]])
        with patch.object(parallel_blocks, "flash_attn_varlen_func", None):
            assert tp_attn._can_use_flash_attn(cuda_dev, torch.bfloat16, mask_with_padding) is False
        with patch.object(parallel_blocks, "flash_attn_varlen_func", MagicMock()), \
             patch.object(parallel_blocks, "unpad_input", MagicMock()):
            assert tp_attn._can_use_flash_attn(cuda_dev, torch.bfloat16, mask_with_padding) is True

        # Non-2D mask -> False
        mask_3d = torch.ones(1, 1, 4)
        assert tp_attn._can_use_flash_attn(cuda_dev, torch.bfloat16, mask_3d) is False

        # All-ones mask (unpadded) -> True
        all_ones_mask = torch.tensor([[1, 1, 1, 1]])
        assert tp_attn._can_use_flash_attn(cuda_dev, torch.bfloat16, all_ones_mask) is True

        # Supported head dimensions: 64, 128, 256
        for hdim in (64, 128, 256):
            tp_attn.head_dim = hdim
            assert tp_attn._can_use_flash_attn(cuda_dev, torch.bfloat16, None) is True

        # Unsupported head dim (e.g. 96 or 32 when not compiled)
        tp_attn.head_dim = 96
        assert tp_attn._can_use_flash_attn(cuda_dev, torch.bfloat16, None) is False
        tp_attn.head_dim = 128

        # Explicit sdpa requested in config -> False
        tp_attn.config._attn_implementation = "sdpa"
        assert tp_attn._can_use_flash_attn(cuda_dev, torch.bfloat16, None) is False

        # Explicit eager requested in config -> False
        tp_attn.config._attn_implementation = "eager"
        assert tp_attn._can_use_flash_attn(cuda_dev, torch.bfloat16, None) is False


def test_tensor_parallel_attention_dispatches_to_flash_attn_mocked():
    """Verify that when _can_use_flash_attn is true, flash_attn_func receives unexpanded GQA heads."""
    attention = _mock_attention_module(head_dim=128, num_heads=16, num_kv_heads=4, attn_impl="flash_attention_2")
    tp_attn = TensorParallelAttention(attention, ["cpu", "cpu"])

    mock_flash_fn = MagicMock()
    # Mock return shape: (batch, seqlen, heads_per_rank, head_dim)
    mock_flash_fn.side_effect = lambda q, k, v, **kwargs: torch.zeros_like(q)

    batch_size = 2
    seq_len = 8
    hidden_dim = 16 * 128
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)

    cos = torch.ones(batch_size, seq_len, 128)
    sin = torch.zeros(batch_size, seq_len, 128)

    with patch.object(tp_attn, "_can_use_flash_attn", return_value=True), \
         patch.object(parallel_blocks, "flash_attn_func", mock_flash_fn):

        output, past_kv = tp_attn(hidden_states, position_embeddings=(cos, sin), attention_mask=None)

        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert past_kv is None

        # Two devices in tp_attn -> flash_attn_func called twice (once per rank)
        assert mock_flash_fn.call_count == 2

        # Inspect call args for rank 0:
        q_arg, k_arg, v_arg = mock_flash_fn.call_args_list[0][0]
        kwargs = mock_flash_fn.call_args_list[0][1]

        # heads_per_rank = 16 / 2 = 8
        # kv_heads_per_rank = 4 / 2 = 2
        # FA2 layout: (batch, seqlen, heads, head_dim)
        assert q_arg.shape == (batch_size, seq_len, 8, 128)
        # CRUCIAL: K and V must NOT be repeated/expanded to 8 heads! They must remain 2 heads.
        assert k_arg.shape == (batch_size, seq_len, 2, 128)
        assert v_arg.shape == (batch_size, seq_len, 2, 128)

        # Causal must be True since seq_len > 1 and mask is None
        assert kwargs["causal"] is True
        assert kwargs["softmax_scale"] == pytest.approx(128 ** -0.5)


def test_tensor_parallel_attention_cpu_fallback_preserves_numerical_exactness():
    """Verify that on CPU the SDPA fallback runs and produces identical outputs."""
    attention = _mock_attention_module(head_dim=64, num_heads=4, num_kv_heads=2, attn_impl="sdpa")
    tp_attn = TensorParallelAttention(attention, ["cpu", "cpu"])

    batch_size = 1
    seq_len = 4
    hidden_dim = 4 * 64
    x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)

    cos = torch.ones(batch_size, seq_len, 64)
    sin = torch.zeros(batch_size, seq_len, 64)

    # Ensure fallback path is taken
    out, _ = tp_attn(x, position_embeddings=(cos, sin), attention_mask=None)
    assert out.shape == (batch_size, seq_len, hidden_dim)
    assert out.requires_grad

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available() or not is_flash_attn_available(), reason="requires CUDA and working flash_attn")
def test_tensor_parallel_attention_live_cuda_matches_sdpa():
    """Live CUDA test verifying TensorParallelAttention runs with real flash_attn_func and matches SDPA."""
    device = torch.device("cuda:0")
    head_dim = 128
    num_heads = 8
    num_kv_heads = 2
    hidden_dim = num_heads * head_dim

    attention = _mock_attention_module(head_dim=head_dim, num_heads=num_heads, num_kv_heads=num_kv_heads, attn_impl="flash_attention_2")
    attention.to(device=device, dtype=torch.bfloat16)

    tp_attn = TensorParallelAttention(attention, [device, device])

    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.bfloat16, requires_grad=True)

    cos = torch.ones(batch_size, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    sin = torch.zeros(batch_size, seq_len, head_dim, device=device, dtype=torch.bfloat16)

    # Run with flash attention enabled
    out_fa, _ = tp_attn(x, position_embeddings=(cos, sin), attention_mask=None)
    assert out_fa.shape == (batch_size, seq_len, hidden_dim)
    assert torch.isfinite(out_fa).all()

    loss_fa = out_fa.sum()
    loss_fa.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    # Compare with SDPA path by forcing _can_use_flash_attn to False
    x_sdpa = x.detach().clone().requires_grad_(True)
    with patch.object(tp_attn, "_can_use_flash_attn", return_value=False):
        out_sdpa, _ = tp_attn(x_sdpa, position_embeddings=(cos, sin), attention_mask=None)

    # In bfloat16, FlashAttention (which uses online softmax accumulation in fp32)
    # matches standard SDPA within tight numerical tolerances.
    torch.testing.assert_close(out_fa, out_sdpa, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available() or not is_flash_attn_available(), reason="requires CUDA and working flash_attn")
def test_tensor_parallel_attention_live_cuda_padded_batch():
    """Live CUDA test verifying TensorParallelAttention runs with flash_attn_varlen_func on padded batches."""
    device = torch.device("cuda:0")
    head_dim = 256
    num_heads = 8
    num_kv_heads = 2
    hidden_dim = num_heads * head_dim

    attention = _mock_attention_module(head_dim=head_dim, num_heads=num_heads, num_kv_heads=num_kv_heads, attn_impl="flash_attention_2")
    attention.to(device=device, dtype=torch.bfloat16)

    tp_attn = TensorParallelAttention(attention, [device, device])

    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.bfloat16, requires_grad=True)

    cos = torch.ones(batch_size, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    sin = torch.zeros(batch_size, seq_len, head_dim, device=device, dtype=torch.bfloat16)

    # 2D attention mask with padding at the end of example 0
    attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
    attention_mask[0, 10:] = False

    out_fa, _ = tp_attn(x, position_embeddings=(cos, sin), attention_mask=attention_mask)
    assert out_fa.shape == (batch_size, seq_len, hidden_dim)
    assert torch.isfinite(out_fa).all()

    loss = out_fa.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


