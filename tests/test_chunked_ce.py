"""Equivalence gate for the chunked cross-entropy.

The whole justification for chunking is that it changes memory, not numbers. These
compare it against the transformers implementation it replaces -- including the
gradient, since the checkpointed recompute is where a chunked version would most
plausibly go wrong.
"""

import pytest
import torch
from transformers.loss.loss_utils import ForCausalLMLoss

from distillkit.chunked_ce import chunked_causal_lm_loss


def _batch(batch, seq, vocab, seed=0, dtype=torch.float32, ignore_frac=0.0):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, seq, vocab, generator=generator, dtype=dtype)
    labels = torch.randint(0, vocab, (batch, seq), generator=generator)
    if ignore_frac:
        drop = torch.rand(batch, seq, generator=generator) < ignore_frac
        labels = labels.masked_fill(drop, -100)
    return logits, labels


@pytest.mark.parametrize("batch,seq,vocab", [(1, 16, 64), (2, 33, 128), (3, 8, 32)])
@pytest.mark.parametrize("chunk", [1, 7, 4096, None])
def test_matches_transformers_value(batch, seq, vocab, chunk):
    logits, labels = _batch(batch, seq, vocab)
    expected = ForCausalLMLoss(logits, labels, vocab_size=vocab)
    actual = chunked_causal_lm_loss(logits, labels, vocab_size=vocab, chunk_tokens=chunk)
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6), (
        f"chunk={chunk}: {actual.item()} != {expected.item()}"
    )


@pytest.mark.parametrize("ignore_frac", [0.3, 0.9])
def test_matches_with_ignored_positions(ignore_frac):
    """Padding is masked with -100; the denominator must count only real tokens."""
    logits, labels = _batch(2, 40, 96, seed=1, ignore_frac=ignore_frac)
    expected = ForCausalLMLoss(logits, labels, vocab_size=96)
    actual = chunked_causal_lm_loss(logits, labels, vocab_size=96, chunk_tokens=9)
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_matches_with_num_items_in_batch():
    """Gradient-accumulation path: sum divided by a caller-supplied denominator."""
    logits, labels = _batch(2, 24, 64, seed=2)
    for denom in (17, torch.tensor(17.0)):
        expected = ForCausalLMLoss(logits, labels, vocab_size=64, num_items_in_batch=denom)
        actual = chunked_causal_lm_loss(
            logits, labels, vocab_size=64, num_items_in_batch=denom, chunk_tokens=5
        )
        assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_gradients_match_through_checkpointed_chunks():
    """The recompute is the risky part -- compare dL/dlogits, not just the scalar."""
    logits, labels = _batch(2, 32, 80, seed=3)

    a = logits.clone().requires_grad_(True)
    ForCausalLMLoss(a, labels, vocab_size=80).backward()

    b = logits.clone().requires_grad_(True)
    chunked_causal_lm_loss(b, labels, vocab_size=80, chunk_tokens=7).backward()

    assert a.grad is not None and b.grad is not None
    assert torch.allclose(a.grad, b.grad, rtol=1e-5, atol=1e-7)


def test_bf16_logits_keep_fp32_precision():
    """Chunking must not sacrifice the upcast transformers does for a reason."""
    logits, labels = _batch(2, 32, 256, seed=4, dtype=torch.bfloat16)
    expected = ForCausalLMLoss(logits, labels, vocab_size=256)
    actual = chunked_causal_lm_loss(logits, labels, vocab_size=256, chunk_tokens=11)
    assert actual.dtype == expected.dtype == torch.float32
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_all_ignored_batch_is_finite_not_nan():
    """A pure-padding batch divides by zero in the naive form."""
    logits = torch.randn(1, 8, 32)
    labels = torch.full((1, 8), -100)
    out = chunked_causal_lm_loss(logits, labels, vocab_size=32)
    assert torch.isfinite(out), out


def test_explicit_shift_labels_are_respected():
    logits, labels = _batch(2, 20, 48, seed=5)
    shift = torch.roll(labels, -1, dims=1)
    expected = ForCausalLMLoss(logits, labels, vocab_size=48, shift_labels=shift)
    actual = chunked_causal_lm_loss(
        logits, labels, vocab_size=48, shift_labels=shift, chunk_tokens=6
    )
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_saves_memory_on_a_wide_vocabulary():
    """The point of the exercise, measured rather than asserted in a comment."""
    vocab, batch, seq = 248_320, 1, 1024
    labels = torch.randint(0, vocab, (batch, seq), device="cuda")

    def peak(loss_fn):
        torch.cuda.empty_cache()
        logits = torch.randn(batch, seq, vocab, device="cuda", dtype=torch.bfloat16)
        logits.requires_grad_(True)
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        loss_fn(logits, labels, vocab_size=vocab).backward()
        used = torch.cuda.max_memory_allocated() - before
        del logits
        return used / 1024**3

    stock = peak(ForCausalLMLoss)
    chunked = peak(chunked_causal_lm_loss)
    print(f"\nstock {stock:.2f} GB -> chunked {chunked:.2f} GB")
    assert chunked < stock * 0.7, f"chunked {chunked:.2f} GB vs stock {stock:.2f} GB"


def test_default_chunk_actually_chunks_a_wide_vocabulary():
    """Regression on the bug the memory test caught: a fixed token count larger than
    the batch means one chunk and zero saving. The default must scale with vocab."""
    from distillkit.chunked_ce import DEFAULT_CHUNK_BYTES, chunk_tokens_for

    wide = chunk_tokens_for(248_320)
    assert wide < 1024, f"a 1024-token batch would not be chunked at all ({wide})"
    assert wide * 248_320 * 4 <= DEFAULT_CHUNK_BYTES
    # A narrow vocabulary should not be chunked into uselessly tiny pieces.
    assert chunk_tokens_for(1_000) > 10_000


def test_install_gate_is_the_code_the_trainer_calls():
    """Exercise the real helper, not a copy of its logic.

    A test that reimplements the condition can pass while the trainer is wrong, which
    is worse than no test: the failure it is meant to catch (silently giving back
    5.6 GB of peak VRAM) is invisible in the numbers.
    """
    from types import SimpleNamespace

    from distillkit.chunked_ce import maybe_install_chunked_loss

    def install(need_model_loss, enabled):
        model = SimpleNamespace(loss_function=ForCausalLMLoss)
        swapped = maybe_install_chunked_loss(
            model, need_model_loss=need_model_loss, enabled=enabled
        )
        return swapped, model.loss_function

    assert install(True, True) == (True, chunked_causal_lm_loss)
    assert install(True, False) == (False, ForCausalLMLoss), "config must be able to opt out"
    assert install(False, True) == (False, ForCausalLMLoss), "no consumer, no reason to swap"

    # A model with no loss_function at all must not gain one.
    bare = SimpleNamespace(loss_function=None)
    assert maybe_install_chunked_loss(bare, need_model_loss=True, enabled=True) is False

    # DDP/DeepSpeed wrappers: the swap must land on the inner module.
    inner = SimpleNamespace(loss_function=ForCausalLMLoss)
    wrapper = SimpleNamespace(module=inner)
    assert maybe_install_chunked_loss(wrapper, need_model_loss=True, enabled=True) is True
    assert inner.loss_function is chunked_causal_lm_loss
