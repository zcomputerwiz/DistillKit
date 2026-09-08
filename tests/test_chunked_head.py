"""Folding the head into the chunk loop must change the number, not just the memory.

The point is to stop materializing `[batch, seq, 248320]` logits and their gradient --
1.89 GiB each at sequence 4096, the allocation in every OOM this project has hit. That
is only worth anything if the loss and its gradients are identical to projecting the
whole sequence first, so these compare against exactly that.
"""

import pytest
import torch

from distillkit.chunked_head import chunked_head_loss
from distillkit.lossfuncs.common import accumulate_over_chunks
from distillkit.lossfuncs.kl import sparse_kl_div_inner

VOCAB, HIDDEN, TOP_K = 512, 32, 8


def _fixture(batch=1, seq=64, seed=0, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(batch, seq, HIDDEN, generator=generator, dtype=dtype)
    head = torch.nn.Linear(HIDDEN, VOCAB, bias=False, dtype=dtype)
    with torch.no_grad():
        head.weight.copy_(torch.randn(VOCAB, HIDDEN, generator=generator, dtype=dtype) * 0.05)
    ids = torch.randint(0, VOCAB, (batch, seq, TOP_K), generator=generator)
    values = torch.log_softmax(torch.randn(batch, seq, TOP_K, generator=generator), -1).to(dtype)
    return hidden, head, ids, values


@pytest.mark.parametrize("chunk", [None, 1, 7, 16, 64, 1000])
def test_matches_projecting_the_whole_sequence_first(chunk):
    hidden, head, ids, values = _fixture()
    reference = accumulate_over_chunks(head(hidden), ids, values, None, None, sparse_kl_div_inner)
    got = chunked_head_loss(hidden, head, ids, values, None, chunk, sparse_kl_div_inner)
    torch.testing.assert_close(got, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("chunk", [None, 8, 16])
def test_gradients_match_through_both_the_head_and_the_hidden_state(chunk):
    """The recompute has to rebuild the projection, not just the reduction."""
    hidden, head, ids, values = _fixture()

    reference_hidden = hidden.clone().requires_grad_(True)
    accumulate_over_chunks(
        head(reference_hidden), ids, values, None, None, sparse_kl_div_inner
    ).backward()
    reference_weight_grad = head.weight.grad.clone()
    head.weight.grad = None

    chunked_hidden = hidden.clone().requires_grad_(True)
    chunked_head_loss(
        chunked_hidden, head, ids, values, None, chunk, sparse_kl_div_inner
    ).backward()

    torch.testing.assert_close(chunked_hidden.grad, reference_hidden.grad, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(head.weight.grad, reference_weight_grad, rtol=1e-5, atol=1e-7)


def test_mask_is_sliced_with_the_chunk():
    hidden, head, ids, values = _fixture(seq=32)
    mask = torch.zeros(1, 32, 1)
    mask[:, :10] = 1.0
    reference = accumulate_over_chunks(head(hidden), ids, values, mask, None, sparse_kl_div_inner)
    for chunk in (1, 3, 8, 32):
        got = chunked_head_loss(hidden, head, ids, values, mask, chunk, sparse_kl_div_inner)
        torch.testing.assert_close(got, reference, rtol=1e-5, atol=1e-6)


def test_padded_head_is_truncated_to_the_signal_vocabulary():
    """The student's head is padded wider than the teacher's signal (248320 vs 248077).

    The trainer truncates the materialized logits; chunking has to do the same per
    chunk, or the log-sum-exp normalizes over columns the teacher never scored.
    """
    hidden, head, ids, values = _fixture()
    true_vocab = VOCAB - 7
    ids = ids.clamp(max=true_vocab - 1)
    reference = accumulate_over_chunks(
        head(hidden)[..., :true_vocab], ids, values, None, None, sparse_kl_div_inner
    )
    got = chunked_head_loss(
        hidden, head, ids, values, None, 16, sparse_kl_div_inner, vocab_size=true_vocab
    )
    torch.testing.assert_close(got, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_never_materializes_the_full_vocabulary_logits():
    """The whole point: peak memory must not contain a [seq, vocab] tensor."""
    seq, vocab = 512, 32_000
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(1, seq, HIDDEN, generator=generator).cuda().requires_grad_(True)
    head = torch.nn.Linear(HIDDEN, vocab, bias=False).cuda()
    ids = torch.randint(0, vocab, (1, seq, TOP_K), generator=generator).cuda()
    values = torch.log_softmax(torch.randn(1, seq, TOP_K, generator=generator), -1).cuda()

    def peak(fn):
        head.weight.grad = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        fn().backward()
        return (torch.cuda.max_memory_allocated() - before) / 1024**2

    materialized = peak(
        lambda: accumulate_over_chunks(head(hidden), ids, values, None, 64, sparse_kl_div_inner)
    )
    folded = peak(
        lambda: chunked_head_loss(hidden, head, ids, values, None, 64, sparse_kl_div_inner)
    )
    full_logits_mb = seq * vocab * 4 / 1024**2
    print(f"\nfull logits {full_logits_mb:.0f} MB | materialized {materialized:.0f} MB "
          f"-> folded {folded:.0f} MB")
    # Materializing costs the logits and their gradient; folding should cost neither.
    assert materialized > full_logits_mb, "test is not measuring the logits allocation"
    assert folded < materialized - full_logits_mb, (
        f"folding saved only {materialized - folded:.0f} MB of the "
        f"{full_logits_mb:.0f} MB logits tensor"
    )
