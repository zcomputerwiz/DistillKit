# Assisted-by: Codex
import sys
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scratch/dense_gr'))
from bounded_cached_prefill import QueryChunkSDPA


def test_chunked_sdpa_preserves_mask_scale_and_query_order():
    torch.manual_seed(42)
    query = torch.randn(2, 3, 19, 8)
    key = torch.randn(2, 3, 23, 8)
    value = torch.randn(2, 3, 23, 12)
    mask = torch.zeros(2, 1, 19, 23)
    causal = torch.arange(23)[None, :] > torch.arange(19)[:, None] + 4
    mask.masked_fill_(causal, -torch.inf)
    mask[0, :, :, :2] = -torch.inf
    reference = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=.17)
    with QueryChunkSDPA(5):
        actual = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=.17)
    torch.testing.assert_close(actual, reference)


def test_implicit_causal_chunks_are_rejected():
    q = torch.randn(1, 1, 9, 4)
    with QueryChunkSDPA(3), pytest.raises(ValueError, match='explicit masking'):
        F.scaled_dot_product_attention(q, q, q, is_causal=True)
