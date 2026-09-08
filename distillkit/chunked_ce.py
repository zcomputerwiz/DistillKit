"""Memory-cheap causal-LM cross-entropy for very wide vocabularies.

Qwen3.5's head is 248,320 wide. Transformers' ``ForCausalLMLoss`` starts with::

    logits = logits.float()

which materializes a full fp32 copy of ``[batch * seq, 248320]`` and keeps it alive
for the backward pass. At batch 3 x seq 1024 that single tensor is 3.05 GB, and it
measured as roughly 54% of all activation memory -- more than the 32-layer backbone.
The upcast itself is not optional: cross-entropy in bf16 over 248k classes loses real
precision, which is why transformers does it.

Chunking makes it cheap without giving up the precision. Cross-entropy with
``reduction="sum"`` is additive over tokens, so summing per-chunk sums and dividing by
the total non-ignored count is the same number as one big reduction. Each chunk is
wrapped in ``torch.utils.checkpoint`` so its fp32 upcast is freed immediately and
recomputed during backward -- the recompute is one softmax over a slice, negligible
next to the model it is attached to.

Swap it in with ``model.loss_function = chunked_causal_lm_loss`` (transformers looks
that attribute up per call), so no loss class or call signature has to change.

Equivalence to the stock implementation is asserted in ``tests/test_chunked_ce.py``.
"""

from __future__ import annotations

from types import MethodType

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = [
    "chunked_causal_lm_loss",
    "DEFAULT_CHUNK_BYTES",
    "chunk_tokens_for",
    "maybe_install_chunked_loss",
]

# Budget the chunk by BYTES, not by token count. The live fp32 tensor is
# tokens x vocab x 4, so a fixed token count silently stops chunking whenever the
# batch is smaller than it: at 4096 tokens and a 1024-token batch there is exactly one
# chunk and the saving is zero. That is a real bug this default is shaped to avoid.
DEFAULT_CHUNK_BYTES = 128 * 1024 * 1024  # 128 MB of fp32 logits per chunk


def chunk_tokens_for(vocab_size: int, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> int:
    """Tokens whose fp32 logits fit in `chunk_bytes`. At least 1."""
    return max(1, chunk_bytes // max(1, vocab_size * 4))


def chunked_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    num_items_in_batch: torch.Tensor | int | None = None,
    ignore_index: int = -100,
    shift_labels: torch.Tensor | None = None,
    chunk_tokens: int | None = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    **kwargs,
) -> torch.Tensor:
    """Drop-in replacement for ``transformers.loss.loss_utils.ForCausalLMLoss``.

    Matches its shifting, ignore-index handling and reduction semantics: mean over
    non-ignored tokens, or sum divided by ``num_items_in_batch`` when that is given.
    """
    if shift_labels is None:
        # Shift so that tokens < n predict n -- same as the stock implementation.
        labels = F.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    if chunk_tokens is None:
        chunk_tokens = chunk_tokens_for(vocab_size, chunk_bytes)

    flat_logits = logits.view(-1, vocab_size)
    flat_labels = shift_labels.view(-1).to(flat_logits.device)

    if num_items_in_batch is None:
        denominator = (flat_labels != ignore_index).sum()
    elif torch.is_tensor(num_items_in_batch):
        denominator = num_items_in_batch.to(flat_logits.device)
    else:
        denominator = torch.tensor(
            num_items_in_batch, device=flat_logits.device, dtype=torch.float32
        )

    def chunk_sum(logits_chunk: torch.Tensor, labels_chunk: torch.Tensor) -> torch.Tensor:
        # The .float() lives only inside this call; checkpointing frees it on return.
        return F.cross_entropy(
            logits_chunk.float(), labels_chunk, ignore_index=ignore_index, reduction="sum"
        )

    total = None
    n_tokens = flat_logits.shape[0]
    for start in range(0, n_tokens, chunk_tokens):
        stop = min(start + chunk_tokens, n_tokens)
        logits_chunk = flat_logits[start:stop]
        labels_chunk = flat_labels[start:stop]
        if logits_chunk.requires_grad:
            # This computation is deterministic; restoring global RNG from worker
            # threads would race with other in-flight checkpoint recomputations.
            part = checkpoint(chunk_sum, logits_chunk, labels_chunk, use_reentrant=False, preserve_rng_state=False)
        else:
            part = chunk_sum(logits_chunk, labels_chunk)
        total = part if total is None else total + part

    if total is None:  # empty batch
        return flat_logits.sum() * 0.0

    # A batch of pure padding has no supervised tokens; returning 0 keeps the step
    # finite instead of emitting NaN from a divide by zero.
    denominator = torch.clamp(denominator.to(total.dtype), min=1.0)
    return total / denominator


def maybe_install_chunked_loss(model, *, need_model_loss: bool, enabled: bool) -> bool:
    """Swap `model.loss_function` for the chunked form when it will actually be used.

    Two gates, for different reasons. `need_model_loss` is False when no configured
    loss reads `student_outputs.loss`, in which case the trainer also withholds labels
    and no cross-entropy runs at all -- swapping would be pointless. `enabled` is the
    run config's opt-out for when VRAM is not the constraint and the ~7% recompute
    cost is not worth 5.6 GB.

    Returns whether the swap happened.
    """
    if not (need_model_loss and enabled):
        return False
    base_model = model.module if hasattr(model, "module") else model
    if getattr(base_model, "loss_function", None) is None:
        return False
    base_model.loss_function = chunked_causal_lm_loss
    return True


def keep_bf16_forward_outputs(model) -> bool:
    """Drop accelerate's blanket fp32 upcast of the forward's outputs.

    ``Accelerator.prepare_model`` wraps a mixed-precision model as::

        model.forward = convert_outputs_to_fp32(autocast_context(model.forward))

    and that wrapper walks the returned structure calling ``tensor.float()`` on every
    bf16/fp16 tensor in it. For a 248,320-wide head that is a 3.79 GiB allocation of
    logits at sequence 4096, plus another 3.79 GiB for its gradient, plus ~1.4 GB more
    when ``output_hidden_states`` returns all 33 states. It is what OOM'd both the 1M
    control arm and the first stage-2 attempt.

    It also defeats the point of ``sparse_chunk_length``: the KL loss chunks precisely
    so that no full-vocabulary fp32 tensor ever exists, and this allocates one before
    the loss is even called.

    Dropping it is not free by itself: the sparse divergences took ``out_dtype`` from
    the logits they were handed, so bf16 logits made them difference log-probabilities
    in bf16. ``lossfuncs.common.divergence_dtype`` restores fp32 there, on chunk-sized
    slices, and because widening bf16 to fp32 is exact the result is bit-identical to
    what the blanket upcast produced -- ``tests/test_bf16_outputs.py`` asserts that at
    rtol=atol=0. Do not remove the wrapper without that half.

    The other consumers never needed it:

    * the hidden-state losses align dtypes explicitly against the projection weight;
    * ``cross_entropy`` reads ``student_outputs.loss``, a scalar the model already
      computed, which this wrapper never touched.

    Autocast itself is left in place: only the output conversion is removed. Returns
    whether a wrapper was found and removed, so callers can stay idempotent.
    """
    from accelerate.utils.operations import ConvertOutputsToFp32

    forward = getattr(model, "forward", None)
    if forward is None:
        return False
    # Bound-method form (accelerate's usual path) and plain-function form both occur;
    # which one depends on whether model.forward had __func__ at prepare time.
    function = getattr(forward, "__func__", forward)
    converter = getattr(function, "__wrapped__", None)
    if not isinstance(converter, ConvertOutputsToFp32):
        return False
    inner = converter.model_forward
    if hasattr(forward, "__func__"):
        model.forward = MethodType(inner, model)
    else:
        model.forward = inner
    return True
