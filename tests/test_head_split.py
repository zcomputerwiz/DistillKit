"""Splitting a fused gated-delta call by heads changes nothing but its memory."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fla kernels are CUDA-only")
def test_head_split_matches_one_call(monkeypatch):
    from distillkit.models.qwen35 import linear_attention_dispatch as dispatch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module

    dispatch.install_device_aware_linear_attention()
    call = module.torch_chunk_gated_delta_rule
    torch.manual_seed(0)
    b, t, h, d = 1, 512, 8, 64
    q, k, v = (torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    g = -torch.rand(b, t, h, device="cuda").float()
    beta = torch.rand(b, t, h, device="cuda").to(torch.bfloat16)
    state = torch.randn(b, h, d, d, device="cuda").float()
    kwargs = dict(g=g, beta=beta, initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True)
    whole, whole_state = call(q, k, v, **kwargs)
    monkeypatch.setattr(dispatch, "HEAD_SPLIT_TOKENS", t * h // 4)
    split, split_state = call(q, k, v, **kwargs)
    torch.testing.assert_close(split, whole)
    torch.testing.assert_close(split_state, whole_state)