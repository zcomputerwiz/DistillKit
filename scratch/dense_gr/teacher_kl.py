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

Batches are made uniform by construction rather than by padding. Padding is the thing that
changes CSA2's routing -- where the indexer's scores tie at the top-k cutoff, which position
wins depends on how wide the row is -- and the architecture refuses it outright. So`grouped`sorts by length, takes neighbours, and truncates them to the shortest, which costs 0.2% of
the corpus at six rows because sorted neighbours are nearly the same length. Widths are
floored to the block size, trading 7.7% of the tokens for 8 distinct shapes instead of 520,
because every distinct shape is a cold Triton autotune and a cold autotune under tensor
parallelism is where the two backward threads race.

The group size is a token budget rather than a document count. Memory follows tokens and
this corpus runs 134 to 1024 of them per document, so a fixed count of six makes both a
768-token batch and a 6144-token one, and only the second decides whether the run fits.
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


# Everything about a capture that has to agree before two of them can be read as one
# corpus. The tokenizer decides what the token ids mean; `top_k` sets the width of the
# kept head and therefore the mass the grouped tail has to account for; the temperature,
# `log_values` and `normalization` are the assumptions the grouped tail is derived under.
# A mismatch in any of them is a silently wrong objective rather than a loud failure,
# which is why this is checked rather than trusted.
SHARED_KEYS = ("tokenizer_hash", "tokenizer_vocab_hash", "vocab_size", "top_k",
               "generation_temperature", "log_values", "normalization",
               "position_alignment", "topk_value_dtype")


class MergedCache:
    """Several captures behind one cache's interface.

    Capturing 5M tokens costs hours of the teacher's time, so a corpus that grows should
    not mean recapturing what is already on disk. `CachedTeacher` touches five things on
    a cache -- `documents`, `manifest`, `document_ids`, `read_document` and `close` --
    so merging is a routing table rather than a format change.

    Document ids are unique within a capture and nothing makes them unique across two.
    A collision is raised rather than resolved, because the two entries would have
    different targets and picking either one silently is the kind of error that shows up
    as a slightly worse number months later.
    """

    def __init__(self, paths):
        self.caches = [OfflineTeacherCache(path) for path in paths]
        first = self.caches[0].manifest
        for path, cache in zip(paths[1:], self.caches[1:]):
            differing = [key for key in SHARED_KEYS
                         if cache.manifest.get(key) != first.get(key)]
            if differing:
                raise ValueError(
                    "%s was captured under different settings than %s and cannot be "
                    "merged with it: %s" % (path, paths[0], ", ".join(
                        "%s %r against %r" % (key, cache.manifest.get(key),
                                              first.get(key))
                        for key in differing)))
        self.manifest = first
        self.documents, self._owner = {}, {}
        for path, cache in zip(paths, self.caches):
            for doc_id, document in cache.documents.items():
                if doc_id in self._owner:
                    raise ValueError("document %r is in both %s and %s"
                                     % (doc_id, self._owner[doc_id][0], path))
                self._owner[doc_id] = (path, cache)
                self.documents[doc_id] = document

    def document_ids(self, split=None):
        return [doc_id for cache in self.caches for doc_id in cache.document_ids(split)]

    def read_document(self, doc_id, **kwargs):
        return self._owner[doc_id][1].read_document(doc_id, **kwargs)

    def close(self):
        for cache in self.caches:
            cache.close()


def first_response(ids, marker):
    """Index of the first token after the first assistant marker, or None.

    Scans the ids for the marker run rather than decoding and searching text: the ids
    are what the model reads, and a decode-and-search would match a marker the tokenizer
    had split differently and never actually emitted.

    The *first* marker, not the last: a multi-turn document has several, and what the
    prefix cap decides is whether any response survives it at all.
    """
    marker = list(marker)
    width = len(marker)
    if not width:
        return None
    for start in range(len(ids) - width + 1):
        if list(ids[start:start + width]) == marker:
            return start + width
    return None


class CachedTeacher:
    """Documents and their top-k targets, shuffled, one at a time.

    Holds the cache open and hands out tensors already on the training device. The
    hidden states are never read: they are 5120 wide against this student's 2048 and
    would need a projection to mean anything, while the top-k logprobs are directly
    comparable and are what the objective actually uses.

    `path` may be one capture or several; several are read as one corpus, after checking
    that they agree about everything the objective depends on.
    """

    def __init__(self, path, split="train", device="cuda", seed=0, min_tokens=2,
                 max_length=None, answer_marker=None, min_answer_tokens=0):
        paths = [path] if isinstance(path, (str, Path)) else list(path)
        self.cache = (OfflineTeacherCache(paths[0]) if len(paths) == 1
                      else MergedCache(paths))
        self.device = device
        self.split = split
        # A prefix, never a window from the middle. Causal context means the first `n`
        # positions of a document see exactly what the teacher saw when it produced their
        # targets, so a prefix is free of the mismatch a mid-document window would carry.
        # The cap exists because the sparse stage records attention, which forces the
        # gathered path, whose selection is O(L^2) per layer.
        #
        # Measured, because guessing at it cost a run. Peak over the four longest
        # documents, forward and both losses and backward, without an optimizer: 14.00 GiB
        # at 1024, 16.06 at 1536, 18.17 at 2048, against a 21.60 GiB allowance. That
        # looks like room at 2048 and is not -- Adam's state adds about 2.4 GiB, which
        # puts a real run at roughly 20.6, and the backward then asks for the lm_head
        # gradient in one 970 MiB block and fails. A short real run at 1536 peaked 20.36,
        # still inside a gigabyte of the ceiling. 1024 leaves about three.
        #
        # `--kl-chunk` is not the lever it looks like: 128 against 256 measured identical
        # to two decimal places, because the chunked head frees each chunk's logits before
        # the next and the peak is set by the quadratic routing instead.
        #
        # The cost is corpus: 74.1% of the tokens at 1024 against 89.3% at 2048.
        self.max_length = max_length
        ids = self.cache.document_ids(split)
        self.ids = [doc_id for doc_id in ids
                    if self.cache.documents[doc_id]["length"] >= min_tokens]
        # Both objectives score every position but the last, so a document's system
        # prompt and user turn are trained on exactly like its answer. That is fine
        # while the answer is in there, and the prefix cap makes it a question: a
        # document whose framing runs past the cap is scored entirely on framing, and
        # the model is trained to reproduce a prompt it will never be asked to produce.
        #
        # Measured on `teacher-cache-5m` at a 1024 cap: 115 of 5,303 train documents
        # (2.17%) and 8 of 295 eval documents, about 3.5% of the scored tokens. The
        # tail is long-context documents -- the prompt runs to 7,131 tokens at the
        # worst -- so no cap that fits in 24 GiB reaches their answers.
        #
        # Off by default because turning it on changes the corpus, and every arm
        # measured so far ran without it. `independent_eval` applies the same rule to
        # the NLL bank under `--min-assistant-tokens`, where 21 of 384 documents at a
        # 512-token window were all prompt.
        self.dropped_all_prompt = 0
        if min_answer_tokens > 0:
            if not answer_marker:
                raise ValueError("min_answer_tokens needs answer_marker: the token run "
                                 "that opens an assistant turn in the capture's own "
                                 "vocabulary")
            before = len(self.ids)
            self.ids = [doc_id for doc_id in self.ids
                        if self._answer_tokens(doc_id, answer_marker) >= min_answer_tokens]
            self.dropped_all_prompt = before - len(self.ids)
        self.generator = np.random.default_rng(seed)
        self.tokens = sum(min(self.cache.documents[doc_id]["length"], max_length or 1 << 30)
                          for doc_id in self.ids)
        self.top_k = int(self.cache.manifest["top_k"])
        # `log_values: True` and `generation_temperature: 1.0` are both pinned by the
        # cache format and checked when the manifest is validated, which is what makes
        # the grouped tail applicable here without a temperature argument.

    def _answer_tokens(self, doc_id, marker):
        """Scored answer positions surviving the prefix cap, for one document."""
        ids = self.cache.read_document(doc_id, tokens_only=True)["input_ids"]
        cap = min(len(ids), self.max_length or len(ids))
        start = first_response(ids[:cap], marker)
        # The last kept position has no ground truth and is not scored, hence `cap - 1`.
        return 0 if start is None else max(0, cap - 1 - start)

    def __len__(self):
        return len(self.ids)

    def epochs(self, count):
        """Yield documents forever, reshuffling between passes."""
        order = []
        while True:
            if not order:
                order = list(self.generator.permutation(len(self.ids)))
            yield self.read(self.ids[order.pop()])

    def widths(self, size, block=None, budget=None):
        """The distinct sequence lengths `grouped` will produce, ascending.

        Every distinct shape is a cold Triton autotune, and a cold autotune under tensor
        parallelism is where the two backward threads race over the autotuner's `nargs`.
        Knowing the set in advance means each one can be warmed single-threaded before
        training starts, instead of discovering them over the first epoch.
        """
        return sorted({width for _, width in self._groups(size, block, budget)})

    def _groups(self, size, block=None, budget=None):
        """Sorted neighbours, grouped so every batch costs about the same.

        `size` is a document count and `budget` is a token count; a budget is the better
        unit, because memory follows tokens rather than documents and this corpus runs
        from 134 to 1024 of them per document. A fixed count of six makes a batch of six
        128-token documents and a batch of six 1024-token documents, and only the second
        decides whether the run fits. With a budget the wide groups get fewer rows and
        the narrow ones more, so the widest batch is no larger than the rest.
        """
        lengths = {doc_id: min(self.cache.documents[doc_id]["length"],
                               self.max_length or 1 << 30) for doc_id in self.ids}
        ordered = sorted(self.ids, key=lambda doc_id: lengths[doc_id])
        groups = []
        start = 0
        while start < len(ordered):
            width = lengths[ordered[start]]
            if block:
                width = (width // block) * block
            if width < 2:
                start += 1
                continue
            rows = size if budget is None else max(1, budget // width)
            group = ordered[start:start + rows]
            if len(group) < rows:
                break
            # The width is the shortest in the group, and the group is sorted, so it is
            # the first one -- already floored above.
            groups.append((group, width))
            start += rows
        return groups

    def grouped(self, size, block=None, budget=None):
        """Yield batches of `size` documents that are already the same length.

        One document per forward leaves the card at a fraction of its throughput, and the
        obvious fix -- pad a batch to its longest member -- is the one thing this
        architecture refuses. CSA2 routes over whole blocks and cannot express "half of
        this block is padding", and even where it could, the indexer's top-k ties at the
        cutoff often enough that a padded row routes differently from the same row alone.

        So the batch is made uniform by construction instead: sort by length, take `size`
        neighbours, and truncate them to the shortest. Neighbours in a sorted order are
        close, so the truncation is small -- and it is a prefix, which causal attention
        makes free of any mismatch with the teacher's cached targets.

        The groups are shuffled between passes; the membership is not, because that is
        what keeps a batch uniform.

        `block` floors each width to a multiple of it, which trades tokens for shapes:
        exact widths give 520 distinct lengths and 3,322,458 tokens, flooring to 128 gives
        8 lengths and 92.3% of them. Eight shapes can be warmed before training; 520
        cannot, and every unwarmed one is a chance for the two backward threads to race
        Triton's autotuner. 128 is also CSA2's block size, so the widths line up with the
        routing rather than cutting across it.
        """
        groups = self._groups(size, block, budget)
        while True:
            for index in self.generator.permutation(len(groups)):
                group, width = groups[index]
                yield self.read_batch(group, width)

    def read_batch(self, doc_ids, width):
        """One batch, every row exactly `width` long, nothing padded."""
        ids, targets, values = [], [], []
        for doc_id in doc_ids:
            record = self.cache.read_document(doc_id, include_hidden_states=False)
            ids.append(np.asarray(record["input_ids"][:width], dtype=np.int64))
            targets.append(np.asarray(record["topk_ids"][:width], dtype=np.int64))
            values.append(np.asarray(record["topk_logprobs"][:width], dtype=np.float32))
        return {"input_ids": torch.from_numpy(np.stack(ids)).to(self.device,
                                                                non_blocking=True),
                "topk_ids": torch.from_numpy(np.stack(targets)).to(self.device,
                                                                   non_blocking=True),
                "topk_logprobs": torch.from_numpy(np.stack(values)).to(self.device,
                                                                       non_blocking=True),
                "doc_id": doc_ids[0]}

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
