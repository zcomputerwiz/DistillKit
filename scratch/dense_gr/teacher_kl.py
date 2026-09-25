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

Batches are uniform without padding. Each document is independently capped and rounded
down to the routing block before checking its retained answer. Equal widths are then
batched, including remainders. Row counts and memory budgets cannot change the retained
prefixes. Both input-token storage and supervised-target counts are reported explicitly.

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
                 max_length=None, answer_marker=None, min_answer_tokens=0, exclude=None):
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
        # Documents known to contain benchmark test questions, removed before anything
        # else sees them, so no plan, sample or budget is built over them. An id that is
        # not in any capture is an error rather than a no-op: an exclusion list for the
        # wrong corpus would otherwise exclude nothing and report success.
        self.excluded = 0
        if exclude:
            exclude = set(exclude)
            unknown = exclude - set(self.cache.documents)
            if unknown:
                raise ValueError("%d excluded ids are in none of these captures, e.g. %s"
                                 % (len(unknown), sorted(unknown)[:3]))
            kept = [doc_id for doc_id in ids if doc_id not in exclude]
            self.excluded = len(ids) - len(kept)
            ids = kept
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
        #
        # The answer position is recorded per document rather than used to filter here,
        # because the length a document is finally scored at is not its cap: `grouped`
        # floors each document independently to the routing block. The
        # check that matters is the one `_groups` makes at the retained length.
        self.min_answer_tokens = min_answer_tokens
        self.answer_start = {}
        self.dropped_all_prompt = 0
        if min_answer_tokens > 0:
            if not answer_marker:
                raise ValueError("min_answer_tokens needs answer_marker: the token run "
                                 "that opens an assistant turn in the capture's own "
                                 "vocabulary")
            before = len(self.ids)
            for doc_id in self.ids:
                self.answer_start[doc_id] = self._answer_start(doc_id, answer_marker)
            self.ids = [doc_id for doc_id in self.ids
                        if self._kept_answer(doc_id, self.cap(doc_id)) >= min_answer_tokens]
            self.dropped_all_prompt = before - len(self.ids)
        self.generator = np.random.default_rng(seed)
        self.tokens = sum(min(self.cache.documents[doc_id]["length"], max_length or 1 << 30)
                          for doc_id in self.ids)
        self.top_k = int(self.cache.manifest["top_k"])
        # `log_values: True` and `generation_temperature: 1.0` are both pinned by the
        # cache format and checked when the manifest is validated, which is what makes
        # the grouped tail applicable here without a temperature argument.

    def sources(self):
        """Document ids grouped by the capture they came from, keyed by its path.

        A merge concatenates its captures, so any prefix of `ids` is drawn from the
        first one until it runs out. That is what an evaluation subset must not do.
        """
        owner = getattr(self.cache, "_owner", None)
        if owner is None:
            return {str(getattr(self.cache, "path", "cache")): list(self.ids)}
        grouped = {}
        for doc_id in self.ids:
            grouped.setdefault(str(owner[doc_id][0]), []).append(doc_id)
        return grouped

    def stratified(self, count, seed=12345):
        """`count` documents spread across every capture, the same ones every time.

        Taking the first N of a merged corpus gives N documents from whichever capture
        was named first: a held-out loss over a chat cache and two code caches would
        have been entirely chat, and adding the code captures would have moved the
        number only through the model. Each capture contributes in proportion to its
        size, drawn by a fixed seed, and the sources are visited in sorted order so
        that reordering the `--teacher-cache` arguments selects the same set.
        """
        groups = self.sources()
        if count < 1:
            raise ValueError("evaluation sample count must be positive")
        total = sum(len(ids) for ids in groups.values())
        if not total:
            raise ValueError("no held-out documents survive filtering")
        count = min(count, total)
        generator = np.random.default_rng(seed)
        picked = []
        for name in sorted(groups):
            ids = sorted(groups[name])
            share = max(1, round(count * len(ids) / total)) if ids else 0
            share = min(share, len(ids))
            order = generator.permutation(len(ids))[:share]
            picked.extend((name, ids[index]) for index in sorted(order))
        # Proportional rounding can overshoot; drop from the largest source first so
        # the small ones keep their representation.
        while len(picked) > count:
            counts = {}
            for name, _ in picked:
                counts[name] = counts.get(name, 0) + 1
            biggest = max(sorted(counts), key=lambda name: counts[name])
            for index in range(len(picked) - 1, -1, -1):
                if picked[index][0] == biggest:
                    picked.pop(index)
                    break
        # Rounding may undershoot too. Fill from the most underrepresented source.
        while len(picked) < count:
            used = {doc for _, doc in picked}
            counts = {name: sum(s == name for s, _ in picked) for name in groups}
            available = [name for name in sorted(groups)
                         if counts[name] < len(groups[name])]
            name = max(available, key=lambda s: count * len(groups[s]) / total - counts[s])
            remaining = sorted(set(groups[name]) - used)
            picked.append((name, remaining[int(generator.integers(len(remaining)))]))
        return picked

    def cap(self, doc_id):
        """Prefix cap before independent block rounding (never neighbor-dependent)."""
        return min(self.cache.documents[doc_id]["length"], self.max_length or 1 << 30)

    def _answer_start(self, doc_id, marker):
        """Index of the first answer token, or None if the document has no answer.

        A document with no chat markup at all -- raw prose, a source file -- is all
        content, so its answer starts at the first token. Without that, a general-text
        corpus has no assistant marker anywhere and every document of it would be
        dropped as "all prompt". The test is the turn opener, the marker's first token:
        a chat document that opens turns but never reaches an assistant turn inside the
        cap still has no answer.
        """
        ids = self.cache.read_document(doc_id, tokens_only=True)["input_ids"][:self.cap(doc_id)]
        start = first_response(ids, marker)
        if start is None and int(marker[0]) not in set(ids.tolist()):
            return 0
        return start

    def _kept_answer(self, doc_id, width):
        """Conservative retained-answer count, preserving the existing filter.

        Counts supervised queries starting at the first response token. The additional
        transition predicting that first response token is deliberately not credited.
        """
        start = self.answer_start.get(doc_id)
        return 0 if start is None else max(0, min(width, self.cap(doc_id)) - 1 - start)

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

    def shapes(self, size, block=None, budget=None):
        """Every distinct `(rows, width)` a pass will forward, ascending.

        `widths` alone is not enough to warm the autotuner, which keys on the whole
        shape. Rows are not constant across groups: a token budget gives each width its
        own row count, and the last group of a width is a remainder that is smaller
        still. Warming a computed row count instead of the observed ones warms kernels
        the run never calls and leaves the ones it does call cold.
        """
        return sorted({(len(group), width)
                       for group, width in self._groups(size, block, budget)})

    def _groups(self, size, block=None, budget=None):
        """Fix each prefix first, filter it once, then batch identical widths.

        `budget` limits input tokens in a forward, not supervised targets. Neither
        it nor the row count may change the prefixes or the retained document set.
        """
        if size < 1 or (block is not None and block < 1) or (budget is not None and budget < 1):
            raise ValueError("batch size, block and budget must be positive")
        buckets = {}
        groups = []
        self.dropped_short = 0
        self.dropped_truncated_answer = 0
        for doc_id in sorted(self.ids):
            width = self.cap(doc_id)
            if block:
                width = (width // block) * block
            if width < 2:
                self.dropped_short += 1
                continue
            if self.min_answer_tokens > 0 and self._kept_answer(doc_id, width) < self.min_answer_tokens:
                self.dropped_truncated_answer += 1
                continue
            if budget is not None and width > budget:
                raise ValueError("micro-token budget is smaller than a retained document")
            buckets.setdefault(width, []).append(doc_id)
        for width, members in sorted(buckets.items()):
            rows = size if budget is None else max(1, budget // width)
            for start in range(0, len(members), rows):
                groups.append((members[start:start + rows], width))
        return groups

    def planned_tokens(self, size, block=None, budget=None):
        """Supervised targets per pass, excluding the final position of each row."""
        return sum(len(group) * (width - 1)
                   for group, width in self._groups(size, block, budget))

    def grouped(self, size, block=None, budget=None):
        """Shuffle canonical equal-width batches, retaining all remainder groups.

        Block rounding bounds the shape set. No group's membership changes the width
        or filtering decision of a document; each prefix remains causally aligned to
        the cached teacher. The resumable trainer uses PlannedBatches for its cursor.
        """
        groups = self._groups(size, block, budget)
        if not groups:
            raise ValueError("no training documents survive the sample plan")
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
                "doc_id": doc_ids[0], "doc_ids": list(doc_ids)}

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


def accumulation_shares(chunks):
    """Each micro-batch's weight in an optimizer step, by the tokens it scores.

    Every term in the objective is a mean over its own scored positions, so averaging
    micro-batch means equally averages means of different denominators: a 134-token
    document and a 1024-token one count the same, and the gradient depends on where the
    accumulation boundaries happened to fall rather than on the examples. Weighting each
    by its share of the step's scored tokens makes an accumulated step equal to one
    forward over the same examples, which is the property that lets a batch size change
    for memory without changing what is learned.

    The last position of a row has no ground truth and is not scored, hence the
    subtraction of the row count.
    """
    counts = [int(chunk.numel() - chunk.shape[0]) for chunk in chunks]
    total = max(sum(counts), 1)
    return counts, total


def scored_mask(length, device, rows=1):
    """The positions both objectives score, for every row in the forward.

    Cross-entropy at position t predicts token t+1, so the last position has no ground
    truth. The teacher has an opinion there, but scoring it under one objective and not
    the other would make the blend weight mean something different at the end of every
    document, so both stop at the same place.

    `rows` exists because the caller divides the summed divergence by `mask.sum()`, and
    a `[1, length]` mask broadcasts over the batch while counting one row of it. The
    summed KL then covers `rows * (length - 1)` positions and the divisor covers
    `length - 1`, so the reported per-token divergence -- and with it the teacher's
    share of the blend -- comes out multiplied by the batch size. Measured on one
    document duplicated into a batch: 2.770803 at one row, 5.541607 at two, 16.624821
    at six, where all three are the same document. Cross entropy is a mean over every
    scored position in the batch, so only the KL moved and `--teacher-weight` stopped
    meaning what it says. Returning the mask at full width makes the count match the
    sum by construction.
    """
    mask = torch.ones(rows, length, dtype=torch.bool, device=device)
    mask[:, -1] = False
    return mask
