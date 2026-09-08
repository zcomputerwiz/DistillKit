"""Qwen3.5's grouped-query heads must not reach SDPA as a broadcast request.

transformers sets `enable_gqa=True` whenever there is no attention mask and
head_dim <= 256, to keep SDPA off the math kernel. That holds where a fused kernel
implements the broadcast. This build has none -- it reports "Torch was not compiled
with flash attention" and the memory-efficient kernel refuses unequal head counts --
so the flag meant to avoid the math kernel is what selects it: 4328 MiB per
attention call at sequence 4096 against 249 MiB with the heads expanded.
"""

import pytest
import torch

from distillkit.gqa_dispatch import fused_kernel_supports_gqa, install_expanded_gqa_attention


def test_patch_is_conditional_on_the_build():
    """A build that gains a GQA-capable fused kernel must keep upstream behaviour."""
    from transformers.integrations import sdpa_attention

    original = sdpa_attention.use_gqa_in_sdpa
    try:
        sdpa_attention.use_gqa_in_sdpa = lambda mask, key, value: True
        import distillkit.gqa_dispatch as module

        module._installed = False
        assert module.install_expanded_gqa_attention(force=False) is False
        assert sdpa_attention.use_gqa_in_sdpa(None, None, None) is True

        module._installed = False
        assert module.install_expanded_gqa_attention(force=True) is True
        assert sdpa_attention.use_gqa_in_sdpa(None, None, None) is False
    finally:
        sdpa_attention.use_gqa_in_sdpa = original
        import distillkit.gqa_dispatch as module

        module._installed = False


def test_install_is_idempotent():
    import distillkit.gqa_dispatch as module

    module._installed = False
    from transformers.integrations import sdpa_attention

    original = sdpa_attention.use_gqa_in_sdpa
    try:
        assert module.install_expanded_gqa_attention(force=True) is True
        patched = sdpa_attention.use_gqa_in_sdpa
        assert module.install_expanded_gqa_attention(force=True) is True
        assert sdpa_attention.use_gqa_in_sdpa is patched
    finally:
        sdpa_attention.use_gqa_in_sdpa = original
        module._installed = False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_expanding_kv_heads_is_dramatically_cheaper_when_no_fused_kernel_broadcasts():
    if fused_kernel_supports_gqa():
        pytest.skip("this build has a fused kernel for broadcast GQA; nothing to fix")

    batch, q_heads, kv_heads, seq, dim = 1, 16, 4, 1024, 256
    query = torch.randn(batch, q_heads, seq, dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(batch, kv_heads, seq, dim, device="cuda", dtype=torch.bfloat16)

    def expand(tensor):
        return (
            tensor[:, :, None]
            .expand(batch, kv_heads, q_heads // kv_heads, seq, dim)
            .reshape(batch, q_heads, seq, dim)
        )

    def peak(k, v, **kwargs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        torch.nn.functional.scaled_dot_product_attention(query, k, v, is_causal=True, **kwargs)
        return (torch.cuda.max_memory_allocated() - before) / 1024**2

    broadcast = peak(key, key, enable_gqa=True)
    expanded = peak(expand(key), expand(key))
    print(f"\nenable_gqa {broadcast:.0f} MiB -> expanded {expanded:.0f} MiB")
    assert expanded < broadcast / 4, (
        f"expanding saved only {broadcast - expanded:.0f} MiB of {broadcast:.0f} MiB"
    )
