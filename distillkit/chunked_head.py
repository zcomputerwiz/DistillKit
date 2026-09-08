"""Fold the output head into the loss's chunk loop.

``sparse_chunk_length`` already splits the KL reduction over positions, but it is
handed logits that ``lm_head`` has *already* materialized: at sequence 4096 over a
248,320-wide vocabulary that is 1.89 GiB, plus 1.89 GiB for its gradient. Those two
tensors are the allocation named in every OOM traceback this project has produced, and
chunking the reduction does nothing about them.

Computing the head inside the loop fixes that. Each chunk projects only its own slice
of the post-norm hidden state, reduces it to a scalar contribution, and frees the
logits before the next chunk starts; ``torch.utils.checkpoint`` recomputes the slice
during backward. At 256 positions a chunk's logits are 127 MB rather than 1.89 GiB,
and no full-vocabulary tensor is ever alive.

The hidden state this needs is the model's post-norm output -- exactly ``lm_head``'s
input, and exactly what the anchor tap already captures for the hidden-state loss at
index ``num_hidden_layers``. So the caller passes ``logits_to_keep=1`` to make the
model's own head nearly free and hands the tapped state here instead.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.utils.checkpoint import checkpoint


def chunked_head_loss(
    hidden_states: torch.Tensor,
    head: torch.nn.Module,
    target_ids: torch.LongTensor,
    target_values: torch.Tensor,
    mask: torch.Tensor | None,
    chunk_length: int | None,
    fn: Callable,
    *args,
    vocab_size: int | None = None,
    **kwargs,
) -> torch.Tensor:
    """Accumulate ``fn`` over position chunks, projecting each chunk through ``head``.

    ``fn`` has the signature the sparse divergences already use --
    ``fn(logits, target_ids, target_values, mask, *args, **kwargs)`` -- so this is a
    drop-in for ``accumulate_over_chunks`` when the caller holds a hidden state rather
    than logits.

    ``vocab_size`` truncates each chunk's logits to the teacher's vocabulary, matching
    the trainer's behaviour when the student's head is padded wider than the signal.
    """
    # The head may live on another card (tensor parallelism parks the tied embedding
    # off the home card). Move its input over once and keep every chunk there.
    hidden_states = hidden_states.to(head.weight.device)
    batch, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    if chunk_length is None:
        chunk_length = seq_len
    else:
        # chunk_length counts positions, but a chunk's logits are
        # [batch, chunk_length, vocab] -- its memory scales with the batch too. Read
        # the configured value as a row budget at batch 1 and divide, or the same
        # setting quietly allocates 4x more at batch 4: [4, 256, 248320] in fp32 is
        # 970 MiB, which is what OOM'd this configuration twice.
        chunk_length = max(1, chunk_length // max(1, batch))

    total = None
    for start in range(0, seq_len, chunk_length):
        end = min(start + chunk_length, seq_len)
        chunk_hidden = hidden_states[:, start:end]
        chunk_ids = target_ids[:, start:end]
        chunk_values = target_values[:, start:end]
        chunk_mask = None if mask is None else mask[:, start:end]

        def compute(hidden, ids, values, current_mask=chunk_mask):
            logits = head(hidden)
            if vocab_size is not None and logits.shape[-1] > vocab_size:
                logits = logits[..., :vocab_size]
            return fn(logits, ids, values, current_mask, *args, **kwargs)

        if chunk_hidden.requires_grad:
            # preserve_rng_state=False: this recompute is deterministic, and restoring
            # global RNG from a worker thread would race other in-flight recomputes.
            part = checkpoint(
                compute, chunk_hidden, chunk_ids, chunk_values,
                use_reentrant=False, preserve_rng_state=False,
            )
        else:
            part = compute(chunk_hidden, chunk_ids, chunk_values)
        total = part if total is None else total + part
    return total


class HeadContext:
    """The head, its input, and the vocabulary to truncate to.

    Carries what a loss needs to project logits for itself. ``hidden_states`` is the
    model's post-norm output -- ``lm_head``'s input -- captured by the anchor tap.
    """

    def __init__(self, hidden_states, head, vocab_size=None, chunk_length=None):
        self.hidden_states = hidden_states
        self.head = head
        self.vocab_size = vocab_size
        self.chunk_length = chunk_length

    @property
    def device(self):
        return self.head.weight.device

    def accumulate(self, fn, target_ids, target_values, mask, *args, **kwargs):
        return chunked_head_loss(
            self.hidden_states, self.head, target_ids, target_values, mask,
            self.chunk_length, fn, *args, vocab_size=self.vocab_size, **kwargs,
        )
