"""`sparse_chunk_length` must actually reduce peak memory during training.

The accumulator sums per-chunk losses. Without checkpointing every chunk's fp32
log_softmax stays in the autograd graph until backward, so peak memory equals the
unchunked case and the knob silently does nothing where it matters -- it only helps
under no_grad. Over a 248,320-wide vocabulary that is several GB.

These pin both halves: the value must be unchanged by chunking, and the memory must
actually drop.
"""

import pytest
import torch

from distillkit.lossfuncs.common import accumulate_over_chunks


def _fake_sparse_loss(logits, target_ids, target_values, mask, temperature=1.0):
    """Stand-in with the shape that matters: a full-vocabulary fp32 log_softmax."""
    logprobs = torch.log_softmax(logits.float() / temperature, dim=-1)
    gathered = logprobs.gather(-1, target_ids)
    per_token = (target_values.exp() * (target_values - gathered)).sum(-1)
    if mask is not None:
        per_token = per_token * mask.squeeze(-1)
    return per_token.sum()


def _inputs(batch=1, seq=64, vocab=512, k=8, seed=0, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(batch, seq, vocab, generator=g).to(device)
    ids = torch.randint(0, vocab, (batch, seq, k), generator=g).to(device)
    values = torch.log_softmax(torch.randn(batch, seq, k, generator=g), dim=-1).to(device)
    return logits, ids, values


@pytest.mark.parametrize("chunk", [None, 1, 7, 16, 64, 1000])
def test_chunking_does_not_change_the_value(chunk):
    logits, ids, values = _inputs()
    ref = accumulate_over_chunks(logits, ids, values, None, None, _fake_sparse_loss)
    got = accumulate_over_chunks(logits, ids, values, None, chunk, _fake_sparse_loss)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("chunk", [None, 8, 16])
def test_chunking_does_not_change_the_gradient(chunk):
    """The checkpointed recompute is where a chunked accumulator would go wrong."""
    logits, ids, values = _inputs()

    a = logits.clone().requires_grad_(True)
    accumulate_over_chunks(a, ids, values, None, None, _fake_sparse_loss).backward()

    b = logits.clone().requires_grad_(True)
    accumulate_over_chunks(b, ids, values, None, chunk, _fake_sparse_loss).backward()

    torch.testing.assert_close(b.grad, a.grad, rtol=1e-5, atol=1e-7)


def test_mask_is_sliced_per_chunk():
    """A mask sliced wrongly would weight the wrong positions and still look finite."""
    logits, ids, values = _inputs(seq=32)
    mask = torch.zeros(1, 32, 1)
    mask[:, :10] = 1.0
    ref = accumulate_over_chunks(logits, ids, values, mask, None, _fake_sparse_loss)
    for chunk in (1, 3, 8, 32):
        got = accumulate_over_chunks(logits, ids, values, mask, chunk, _fake_sparse_loss)
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_chunking_actually_reduces_peak_memory_when_training():
    """The point of the knob. Fails if the accumulator stops checkpointing."""
    vocab, seq = 32_000, 512

    def peak(chunk):
        torch.cuda.empty_cache()
        logits, ids, values = _inputs(seq=seq, vocab=vocab, device="cuda")
        logits.requires_grad_(True)
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        accumulate_over_chunks(logits, ids, values, None, chunk, _fake_sparse_loss).backward()
        used = torch.cuda.max_memory_allocated() - before
        del logits, ids, values
        return used / 1024**2

    unchunked = peak(None)
    chunked = peak(64)
    # The logits tensor and its gradient are retained either way; chunking can only
    # remove the transient fp32 log_softmax on top of that floor, so measure against
    # the floor rather than against the total.
    floor = 2 * seq * vocab * 4 / 1024**2
    print(f"\nfloor {floor:.0f} MB | unchunked {unchunked:.0f} MB -> chunked {chunked:.0f} MB")
    assert unchunked > floor, "test is not measuring any reducible memory"
    reducible_before = unchunked - floor
    reducible_after = max(chunked - floor, 0.0)
    assert reducible_after < reducible_before * 0.35, (
        f"chunking freed only {reducible_before - reducible_after:.0f} MB of "
        f"{reducible_before:.0f} MB reducible (floor {floor:.0f} MB)"
    )
