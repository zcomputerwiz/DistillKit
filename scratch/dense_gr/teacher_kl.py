"""Distil the converted model against the cached teacher, one document per forward.

The conversion screen left two regressions with different causes. MMLU fell 12.5 points to
the conversion itself and 30M tokens of plain cross-entropy recovered only 2.7 of them.
Control tokens survived the conversion and were then destroyed by that same training,
because the corpus it ran on -- code and prose -- carries essentially none of this
tokenizer's protocol tokens and nothing held their rows in place.

Both point here. The capture in ``teacher-cache-5m`` is chat-formatted throughout, carries
the reasoning and multiple-choice shapes MMLU asks for, and comes with a 27B teacher's
top-64 distribution at every position. Its documents were excluded from the evaluation
bundle when that bundle was built, so training on it leaves the screen honest -- checked,
not assumed: 768 held-out documents against 5659 cached, overlap 0.

Two decisions are load-bearing.

**The tail is carried, not deleted.** ``MissingProbabilityHandling.ZERO`` -- the default,
and what every config in this tree used -- renormalises the cached top-k to sum to one, so
every omitted token gets target probability exactly zero and a student that puts mass on a
true token the teacher never ranked is penalised for being right. Measured on this cache
the top-64 covers 0.9933 of the mass on average, but 0.4% of positions have more than half
of it outside, and those are exactly the positions where the teacher knew it was uncertain.
``SYMMETRIC_UNIFORM`` distils over the k cached tokens plus one bucket carrying the rest;
the per-token factor cancels between teacher and student, so it asserts only the mass the
cache records. It is only defined at the capture temperature, which is why nothing here
takes a temperature argument.

**Ground truth stays.** Repairing the tail alone was measured and changed nothing: the tail
carried 32.2% of the harm, so two thirds of it sat inside the teacher's list where grouping
has no effect. What removed the harm was keeping cross-entropy alongside, at which point the
penalty fell 38-fold to an interval spanning zero. So this is a blend, not a replacement --
and the ground-truth half is also the only term that says anything about a control token the
teacher's top-64 did not rank.

One document per forward, at its own length. The alternative is a padded batch, and padding
is the thing that changes CSA2's routing: where the indexer's scores tie at the top-k
cutoff, which position wins depends on how wide the row is. Fixed-length windows would
avoid that too, but this corpus is median 537 tokens and a 1024-token window would throw
away 71% of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from distillkit.core.chunked_head import chunked_head_loss
from distillkit.lossfuncs.kl import sparse_kl_div_inner
from distillkit.missing_probability import MissingProbabilityHandling
from distillkit.offline_cache import OfflineTeacherCache


class CachedTeacher:
    """Documents and their top-k targets, shuffled, one at a time.

    Holds the cache open and hands out tensors already on the training device. The
    hidden states are never read: they are 5120 wide against this student's 2048 and
    would need a projection to mean anything, while the top-k logprobs are directly
    comparable and are what the objective actually uses.
    """

    def __init__(self, path, split="train", device="cuda", seed=0, min_tokens=2,
                 max_length=None):
        self.cache = OfflineTeacherCache(path)
        self.device = device
        self.split = split
        # A prefix, never a window from the middle. Causal context means the first `n`
        # positions of a document see exactly what the teacher saw when it produced their
        # targets, so a prefix is free of the mismatch a mid-document window would carry.
        # The cap exists because the sparse stage records attention, which forces the
        # gathered path, whose selection is O(L^2) per layer: a 4096-token document OOMs
        # 24 GiB. At 2048 this keeps 89.3% of the corpus, at 1024 74.1%.
        self.max_length = max_length
        ids = self.cache.document_ids(split)
        self.ids = [doc_id for doc_id in ids
                    if self.cache.documents[doc_id]["length"] >= min_tokens]
        self.generator = np.random.default_rng(seed)
        self.tokens = sum(min(self.cache.documents[doc_id]["length"], max_length or 1 << 30)
                          for doc_id in self.ids)
        self.top_k = int(self.cache.manifest["top_k"])
        # `log_values: True` and `generation_temperature: 1.0` are both pinned by the
        # cache format and checked when the manifest is validated, which is what makes
        # the grouped tail applicable here without a temperature argument.

    def __len__(self):
        return len(self.ids)

    def epochs(self, count):
        """Yield documents forever, reshuffling between passes."""
        order = []
        while True:
            if not order:
                order = list(self.generator.permutation(len(self.ids)))
            yield self.read(self.ids[order.pop()])

    def read(self, doc_id):
        record = self.cache.read_document(doc_id, include_hidden_states=False)
        end = self.max_length or len(record["input_ids"])
        ids = torch.from_numpy(
            np.asarray(record["input_ids"][:end], dtype=np.int64)).unsqueeze(0)
        target_ids = torch.from_numpy(
            np.asarray(record["topk_ids"][:end], dtype=np.int64)).unsqueeze(0)
        # fp16 log probabilities widen to fp32 exactly; the divergence has no business
        # being differenced at fp16 spacing over a 248320-wide vocabulary.
        values = torch.from_numpy(
            np.asarray(record["topk_logprobs"][:end], dtype=np.float32)).unsqueeze(0)
        return {"input_ids": ids.to(self.device, non_blocking=True),
                "topk_ids": target_ids.to(self.device, non_blocking=True),
                "topk_logprobs": values.to(self.device, non_blocking=True),
                "doc_id": doc_id}

    def close(self):
        self.cache.close()


def grouped_tail_kl(hidden, head, target_ids, target_values, mask, chunk_length=256):
    """KL against the cached top-k plus one bucket for everything it omits.

    Returns the summed divergence over masked positions; the caller divides by the count
    it wants to report per token. The head is projected inside the chunk loop rather than
    beforehand, because a full row here is 248320 wide and the whole point of a chunked
    reduction is lost if the tensor it reduces has already been allocated.
    """
    return chunked_head_loss(
        hidden, head, target_ids, target_values, mask, chunk_length,
        sparse_kl_div_inner,
        missing=MissingProbabilityHandling.SYMMETRIC_UNIFORM,
        log_target=True,
    )


def scored_mask(length, device):
    """The positions both objectives score.

    Cross-entropy at position t predicts token t+1, so the last position has no ground
    truth. The teacher has an opinion there, but scoring it under one objective and not
    the other would make the blend weight mean something different at the end of every
    document, so both stop at the same place.
    """
    mask = torch.ones(1, length, dtype=torch.bool, device=device)
    mask[:, -1] = False
    return mask
