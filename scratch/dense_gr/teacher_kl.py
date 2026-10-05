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


def last_response(ids, marker):
    """Index of the first token after the *last* assistant marker, or None: where a
    rollout's own turn starts, after any earlier turns its prompt carried."""
    marker = list(marker)
    width = len(marker)
    for start in range(len(ids) - width, -1, -1):
        if list(ids[start:start + width]) == marker:
            return start + width
    return None


def assistant_tokens(ids, marker, close):
    """Tokens inside assistant turns: after each `marker` run, through its `close` token.

    An agent trace is mostly what the model never writes -- a harness system prompt of
    thousands of tokens, repeated in every trace, and tool output. Scoring those trains
    the student to recite boilerplate and predict tool results; ARTIST (arXiv 2505.01441)
    masks tool output for the same reason. The turn's own close is the model's to write,
    so it stays in."""
    ids = np.asarray(ids)
    inside = np.zeros(len(ids), dtype=bool)
    marker = np.asarray(marker)
    width, at = len(marker), 0
    while at <= len(ids) - width:
        if ids[at] == marker[0] and np.array_equal(ids[at:at + width], marker):
            start = at + width
            closes = np.nonzero(ids[start:] == close)[0]
            end = start + int(closes[0]) + 1 if len(closes) else len(ids)
            inside[start:end] = True
            at = end
        else:
            at += 1
    return inside


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
                 max_length=None, answer_marker=None, min_answer_tokens=0, exclude=None,
                 suppress=None, kl_only=None, strip_prefix=None, unlikelihood=None,
                 think_close=None, repeat=None, strip_nonthinking=None, ce_only=None,
                 assistant_only=None, turn_close=None, answer_spans=None, answer_weight=1.0):
        paths = [path] if isinstance(path, (str, Path)) else list(path)
        if unlikelihood and (think_close is None or min_answer_tokens <= 0):
            raise ValueError("unlikelihood needs think_close (the `</think>` id) and "
                             "min_answer_tokens > 0, which records where answers start")
        self.think_close = think_close
        self.answer_marker = answer_marker
        # Text the student should not see: the teacher template's injected "Reasoning
        # effort is set to xhigh" system text, which the student's template, the
        # evaluations and serving never produce. `strip_prefix` is a list of
        # `(pattern, begin, end)`: a document opening with the token ids `pattern` loses
        # tokens [begin, end). Every remaining position keeps its target -- the teacher's
        # view with the deleted text as context, a context distillation of it -- except
        # position begin - 1, whose target was for a deleted token; it becomes the token
        # that now follows it.
        self.strip_prefix = [(np.asarray(pattern, dtype=np.int64), begin, end)
                             for pattern, begin, end in (strip_prefix or [])]
        self.offset = {}
        # Token ids to take out of the teacher's answer-region targets; see
        # `suppress_teacher_tokens`. Needs each document's answer start, which only the
        # answer filter records.
        if suppress is not None and min_answer_tokens <= 0:
            raise ValueError("suppress needs min_answer_tokens > 0 so answer starts are known")
        self.suppress = None if suppress is None else np.asarray(suppress, dtype=np.int64)
        # Pad each row up to the block multiple instead of cutting the document down to it
        # (set by the trainer); the padded positions are masked out of every loss.
        self.pad_blocks = False
        self.real_width = {}
        self.suppressed_mass = 0.0
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
        # `strip_nonthinking = (marker, empty)` limits stripping to documents whose final
        # assistant turn (after the ids `marker`) opens with the ids `empty`, an empty think
        # block, or that have no assistant turn: the teacher template injects the effort
        # text only in thinking mode, yet the 5m, expand-code and expand-chat captures were
        # rendered in thinking mode around non-thinking replies, so they carry it where the
        # served template never would. Stripping thinking documents too raised held-out
        # code NLL (ablate_continuation.ps1); this renders every document as served.
        if self.strip_prefix:
            for doc_id in ids:
                tokens = np.asarray(self.cache.read_document(doc_id, tokens_only=True)
                                    ["input_ids"], dtype=np.int64)
                if strip_nonthinking is not None:
                    marker, empty = strip_nonthinking
                    turn = last_response(tokens, marker)
                    if turn is not None and list(tokens[turn:turn + len(empty)]) != list(empty):
                        continue
                for pattern, begin, end in self.strip_prefix:
                    if np.array_equal(tokens[:len(pattern)], pattern):
                        self.offset[doc_id] = (begin, end)
                        break
        self.ids = [doc_id for doc_id in ids
                    if self._length(doc_id) >= min_tokens]
        # Captures of the student's own generations, trained toward the teacher alone:
        # cross entropy on them would reinforce exactly the loops and slips on-policy
        # distillation exists to correct. Batches never mix these with ordinary documents.
        owner = getattr(self.cache, "_owner", None)

        def from_captures(named, what):
            chosen = {str(Path(p).resolve()) for p in (named or [])}
            unknown = chosen - {str(Path(p).resolve()) for p in paths}
            if unknown:
                raise ValueError("%s names captures that are not being read: %s"
                                 % (what, sorted(unknown)))
            return {doc_id for doc_id in self.ids
                    if (str(Path(owner[doc_id][0]).resolve()) if owner
                        else str(Path(paths[0]).resolve())) in chosen}

        # Looping rollouts. The teacher cannot say "stop": read at a looping prefix it
        # predicts more of the loop (~0.9 on the repeated token, `onpolicy_signal.py`),
        # so KL there trains the loop in. Their repeated spans get unlikelihood instead
        # (Welleck et al. 2019) and no KL; the rest of them is KL-only like any rollout.
        self.unlikelihood_ids = from_captures(unlikelihood, "unlikelihood")
        self.kl_only_ids = from_captures(kl_only, "kl_only") | self.unlikelihood_ids
        self.ce_only_ids = from_captures(ce_only, "ce_only")
        if self.ce_only_ids & self.kl_only_ids:
            raise ValueError("a capture cannot be both ce_only and kl_only")
        # Agent traces: only the assistant's own turns are scored (`assistant_tokens`).
        self.assistant_only_ids = from_captures(assistant_only, "assistant_only")
        if self.assistant_only_ids and (not answer_marker or turn_close is None):
            raise ValueError("assistant_only needs answer_marker and turn_close (the <|im_end|> id)")
        self.turn_close = turn_close
        # Structural answer spans (capture_inputs.py `answer_spans`: [start, stop) the body,
        # `stop` its <|im_end|>) in the student's coordinates, i.e. after any stripped span.
        # Their positions weigh `answer_weight` in every loss, the rest of the document 1
        # (0 in assistant-only captures). They also say where a document's answer starts,
        # which a marker scan gets wrong when source code spells chat markup.
        if answer_weight <= 0:
            raise ValueError("answer_weight must be positive")
        self.answer_weight = float(answer_weight)
        self.spans = {}
        for doc_id in self.ids:
            spans = (answer_spans or {}).get(doc_id)
            if not spans:
                continue
            begin, end = self.offset.get(doc_id, (0, 0))
            shift = lambda p: p if p < begin else p - (end - begin)
            if any(begin <= p < end for span in spans for p in span):
                raise ValueError("an answer span of %s overlaps its stripped prefix" % doc_id)
            self.spans[doc_id] = [(shift(a), shift(b)) for a, b in spans]
        self._weight_sums = {}
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
                self.answer_start[doc_id] = (self.spans[doc_id][0][0] if doc_id in self.spans
                                             else self._answer_start(doc_id, answer_marker))
            self.ids = [doc_id for doc_id in self.ids
                        if self._kept_answer(doc_id, self.cap(doc_id)) >= min_answer_tokens]
            self.dropped_all_prompt = before - len(self.ids)
        # Whole-number upsampling per capture: a document listed n times is planned n
        # times a pass. The on-policy rounds lost HumanEval+ as math took a larger share
        # of the mix (round 3, with fewer code rollouts, lost most), so the mix is a lever.
        if repeat:
            factor = {str(Path(p).resolve()): int(n) for p, n in repeat.items()}
            unknown = set(factor) - {str(Path(p).resolve()) for p in paths}
            if unknown or min(factor.values()) < 1:
                raise ValueError("repeat needs captures being read and factors >= 1: %s" % repeat)
            self.ids = [doc_id for doc_id in self.ids for _ in range(factor.get(
                str(Path(owner[doc_id][0]).resolve()) if owner else str(Path(paths[0]).resolve()), 1))]
        self.generator = np.random.default_rng(seed)
        self.tokens = sum(self.cap(doc_id) for doc_id in self.ids)
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

    def _length(self, doc_id):
        """The document's length as the student reads it, after any stripped prefix."""
        begin, end = self.offset.get(doc_id, (0, 0))
        return self.cache.documents[doc_id]["length"] - (end - begin)

    def cap(self, doc_id):
        """Prefix cap before independent block rounding (never neighbor-dependent)."""
        return min(self._length(doc_id), self.max_length or 1 << 30)

    def _record(self, doc_id, width=None, **kwargs):
        """The capture's arrays for `doc_id` as the student reads them: any stripped span
        deleted, `width` positions long (all of them when None)."""
        record = self.cache.read_document(doc_id, **kwargs)
        keys = [key for key in ("input_ids", "topk_ids", "topk_logprobs") if key in record]
        if doc_id not in self.offset:
            return {key: record[key][:width] for key in keys}
        begin, end = self.offset[doc_id]
        out = {key: np.concatenate([record[key][:begin], record[key][end:]]) for key in keys}
        if begin > 0 and "topk_ids" in out:
            # The one target that pointed into the deleted span: now the actual next
            # token, with all its mass, so KL there is cross entropy on the kept text.
            out["topk_ids"][begin - 1] = out["input_ids"][begin]
            out["topk_logprobs"][begin - 1] = -1e4
            out["topk_logprobs"][begin - 1, 0] = 0.0
        return {key: value[:width] for key, value in out.items()}

    def _answer_start(self, doc_id, marker):
        """Index of the first answer token, or None if the document has no answer.

        A document with no chat markup at all -- raw prose, a source file -- is all
        content, so its answer starts at the first token. Without that, a general-text
        corpus has no assistant marker anywhere and every document of it would be
        dropped as "all prompt". The test is the turn opener, the marker's first token:
        a chat document that opens turns but never reaches an assistant turn inside the
        cap still has no answer.
        """
        ids = self._record(doc_id, self.cap(doc_id), tokens_only=True)["input_ids"]
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
        copies = {}
        self.dropped_short = 0
        self.dropped_truncated_answer = 0
        for doc_id in sorted(self.ids):
            width = real = self.cap(doc_id)
            if block:
                width = -(-width // block) * block if self.pad_blocks else (width // block) * block
            real = min(real, width)
            if real < 2:
                self.dropped_short += 1
                continue
            self.real_width[doc_id] = real
            if self.min_answer_tokens > 0 and self._kept_answer(doc_id, real) < self.min_answer_tokens:
                self.dropped_truncated_answer += 1
                continue
            if budget is not None and width > budget:
                raise ValueError("micro-token budget is smaller than a retained document")
            copies[doc_id] = copy = copies.get(doc_id, -1) + 1
            buckets.setdefault((width, doc_id in self.kl_only_ids, doc_id in self.unlikelihood_ids,
                                doc_id in self.ce_only_ids), []).append((copy, doc_id))
        for (width, _, _, _), members in sorted(buckets.items()):
            # A repeated document's copies fill successive rounds of its bucket: side by side
            # they would share a microbatch, one visit weighted n times, not n visits.
            members = [doc_id for _, doc_id in sorted(members)]
            rows = size if budget is None else max(1, budget // width)
            for start in range(0, len(members), rows):
                groups.append((members[start:start + rows], width))
        return groups

    def position_weight(self, doc_id, tokens, real):
        """Loss weight of each position of a document's first `real` tokens.

        Position t predicts token t + 1. A document's positions weigh 1 (assistant-only
        captures: 1 in the assistant's turns, 0 elsewhere), its answer spans
        `answer_weight` -- the answer body and its closing `<|im_end|>` -- and its last
        position, which predicts nothing, 0. Padding past `real` weighs nothing.
        """
        weight = np.zeros(real, dtype=np.float32)
        if doc_id in self.assistant_only_ids:
            weight[:-1] = assistant_tokens(tokens[:real], self.answer_marker, self.turn_close)[1:]
        else:
            weight[:-1] = 1.0
        for start, stop in self.spans.get(doc_id, ()):
            first, last = max(start - 1, 0), min(stop, real - 1)
            if first < last:
                weight[first:last] = self.answer_weight
        return weight

    def weight_identity(self):
        """A digest of everything the loss weights depend on beyond the groups."""
        import hashlib
        import json

        payload = dict(answer_weight=self.answer_weight, pad_blocks=self.pad_blocks,
                       spans=sorted((d, [[int(p) for p in s] for s in spans]) for d, spans in self.spans.items()),
                       assistant_only=sorted(self.assistant_only_ids),
                       offsets=sorted((d, [int(p) for p in span]) for d, span in self.offset.items()),
                       marker=[int(t) for t in self.answer_marker or []],
                       close=None if self.turn_close is None else int(self.turn_close))
        return hashlib.sha256(json.dumps(payload).encode()).hexdigest()

    def weighted(self, doc_id):
        """Whether a document needs a weight row (else every position weighs 1)."""
        return doc_id in self.assistant_only_ids or doc_id in self.spans

    def doc_weight(self, doc_id, real):
        """Total loss weight of a document at `real` tokens: what it adds to the budget."""
        if not self.weighted(doc_id):
            return float(real - 1)
        key = (doc_id, real)
        if key not in self._weight_sums:
            tokens = self._record(doc_id, real, tokens_only=True)["input_ids"]
            self._weight_sums[key] = float(self.position_weight(doc_id, tokens, real).sum())
        return self._weight_sums[key]

    def group_weight(self, group, width):
        return sum(self.doc_weight(doc_id, min(self.real_width.get(doc_id, width), width)) for doc_id in group)

    def planned_tokens(self, size, block=None, budget=None):
        """Weighted targets per pass: the sum of every position's loss weight, as trained
        and as the budget counts them (assistant-only rows count their turns, padding
        nothing, answers `answer_weight` each)."""
        return sum(self.group_weight(group, width)
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
        """One batch, every row `width` long: a document's own prefix, then (with
        `pad_blocks`) masked padding up to the block multiple."""
        ids, targets, values, repeats, reals = [], [], [], [], []
        for doc_id in doc_ids:
            real = min(self.real_width.get(doc_id, width), width)
            reals.append(real)
            record = self._record(doc_id, real, include_hidden_states=False)
            ids.append(np.asarray(record["input_ids"], dtype=np.int64))
            targets.append(np.asarray(record["topk_ids"], dtype=np.int64))
            values.append(np.asarray(record["topk_logprobs"], dtype=np.float32))
            if self.suppress is not None:
                start = self.answer_start.get(doc_id, 0)
                # Hedges are an answer's habit: a document with no chat turns at all (raw
                # code, prose) has no answer to keep them out of, and its "Wait" is text.
                chat = self.answer_marker is None or bool((ids[-1] == self.answer_marker[0]).any())
                if start is not None and chat:
                    values[-1], removed = suppress_teacher_tokens(
                        ids[-1], targets[-1], values[-1], self.suppress, start)
                    self.suppressed_mass += removed
            if doc_id in self.unlikelihood_ids:
                turn = last_response(ids[-1], self.answer_marker)
                repeats.append(loop_tokens(ids[-1], len(ids[-1]) if turn is None else turn,
                                           self.think_close))
        if any(real < width for real in reals):
            # Causal: padding after a document changes none of its positions. The pads
            # repeat the last token and carry a uniform, finite teacher row, all masked.
            k = targets[0].shape[1]
            for row, real in enumerate(reals):
                pad = width - real
                if pad:
                    ids[row] = np.concatenate([ids[row], np.full(pad, ids[row][-1], dtype=np.int64)])
                    targets[row] = np.concatenate([targets[row], np.zeros((pad, k), dtype=np.int64)])
                    values[row] = np.concatenate([values[row], np.full((pad, k), -np.log(k), dtype=np.float32)])
                    if repeats:  # unlikelihood documents share a bucket: one entry a row
                        repeats[row] = np.concatenate([repeats[row], np.zeros(pad, dtype=bool)])
        batch = {"input_ids": torch.from_numpy(np.stack(ids)).to(self.device,
                                                                 non_blocking=True),
                 "topk_ids": torch.from_numpy(np.stack(targets)).to(self.device,
                                                                    non_blocking=True),
                 "topk_logprobs": torch.from_numpy(np.stack(values)).to(self.device,
                                                                        non_blocking=True),
                 "doc_id": doc_ids[0], "doc_ids": list(doc_ids),
                 "kl_only": doc_ids[0] in self.kl_only_ids,
                 "ce_only": doc_ids[0] in self.ce_only_ids}
        if any(self.weighted(doc_id) for doc_id in doc_ids) or any(r < width for r in reals):
            # Per-position loss weights (`position_weight`), zero over padding.
            weight = np.zeros((len(ids), width), dtype=np.float32)
            for row, (doc_id, tokens, real) in enumerate(zip(doc_ids, ids, reals)):
                weight[row, :real] = self.position_weight(doc_id, tokens, real)
            batch["weight"] = torch.from_numpy(weight).to(self.device, non_blocking=True)
        if repeats:
            # Position t predicts token t + 1, so a repeated token at t + 1 is a
            # negative at t; the last position predicts nothing.
            negative = np.zeros((len(ids), width), dtype=bool)
            negative[:, :-1] = np.stack(repeats)[:, 1:]
            batch["negative"] = torch.from_numpy(negative).to(self.device, non_blocking=True)
        return batch

    def read(self, doc_id):
        record = self._record(doc_id, self.max_length, include_hidden_states=False)
        ids = torch.from_numpy(
            np.asarray(record["input_ids"], dtype=np.int64)).unsqueeze(0)
        target_ids = torch.from_numpy(
            np.asarray(record["topk_ids"], dtype=np.int64)).unsqueeze(0)
        # fp16 log probabilities widen to fp32 exactly; the divergence has no business
        # being differenced at fp16 spacing over a 248320-wide vocabulary.
        values = torch.from_numpy(
            np.asarray(record["topk_logprobs"], dtype=np.float32)).unsqueeze(0)
        return {"input_ids": ids.to(self.device, non_blocking=True),
                "topk_ids": target_ids.to(self.device, non_blocking=True),
                "topk_logprobs": values.to(self.device, non_blocking=True),
                "doc_id": doc_id}

    def close(self):
        self.cache.close()


def suppress_teacher_tokens(ids, topk_ids, topk_logprobs, suppressed, start):
    """The teacher's distribution with `suppressed` tokens removed, where the text disagrees.

    The teacher only scores text, but a thinking model's scores still say how it would go
    on: on code answers that never hedge, the 27B puts 0.69% of every line start on "Wait",
    "Actually" or "Hmm". KL toward that teaches a 2B to second-guess itself, which it then
    cannot resolve. So in the answer, from `start` on, a row whose next *actual* token is not
    one of `suppressed` loses those entries, and the rest -- top-k and tail alike -- is
    divided by what is left, which keeps their proportions. A row where the text itself
    hedges keeps the teacher's view whole, so the target never contradicts the text.

    Returns the new log probabilities and the total probability mass removed.
    """
    rows = len(ids)
    active = np.zeros(rows, dtype=bool)
    first = max(int(start) - 1, 0)
    if rows > 1 and first < rows - 1:
        active[first:rows - 1] = ~np.isin(ids[first + 1:rows], suppressed)
    drop = np.isin(topk_ids, suppressed) & active[:, None]
    if not drop.any():
        return topk_logprobs, 0.0
    probabilities = np.exp(topk_logprobs.astype(np.float64))
    removed = np.where(drop, probabilities, 0.0).sum(axis=1).clip(max=0.999)
    values = topk_logprobs.astype(np.float64) - np.log1p(-removed)[:, None]
    # The cache's own sentinel for "no probability": finite, so exp() is exactly 0 and
    # nothing downstream multiplies an infinity by zero.
    values[drop] = -1e4
    return values.astype(np.float32), float(removed.sum())


def repeated_tokens(ids, start, n=16, count=3):
    """Tokens from `start` on that belong to the `count`-th or later occurrence of an n-gram.

    One repeat is often legitimate -- the code from the thinking copied into the answer,
    an equation restated -- and an 8-gram, second-occurrence rule flags 39% of rollouts
    that finished cleanly. A loop is the same stretch coming back again and again. Every
    token of a matching n-gram is marked, not only its last, so a loop is penalized from
    the token where it starts copying rather than n - 1 tokens in."""
    marked = np.zeros(len(ids), dtype=bool)
    seen = {}
    for end in range(max(start, 0) + n, len(ids) + 1):
        gram = tuple(ids[end - n:end].tolist())
        seen[gram] = seen.get(gram, 0) + 1
        if seen[gram] >= count:
            marked[end - n:end] = True
    return marked


def loop_tokens(ids, start, close):
    """`repeated_tokens` over the stretch where a repeat means a loop.

    With a thought, that is the thought alone: an answer restating the result the thinking
    checked -- "49413 - 46817 = 2596" a third time -- is the answer, and pushing it down
    teaches the model not to state what it verified. Without one (a non-thinking reply,
    whose think block is empty) it is the whole reply."""
    after = np.nonzero(np.asarray(ids[start:]) == close)[0]
    end = start + int(after[0]) if len(after) else len(ids)
    if end - start <= 4:  # "<think>\n\n</think>": no thought
        return repeated_tokens(ids, start)
    marked = repeated_tokens(ids[:end], start)
    return np.concatenate([marked, np.zeros(len(ids) - end, dtype=bool)])


def loop_start(ids, start, close, n=16, count=3):
    """The token where a loop began copying, or None; over the stretch `loop_tokens` reads.

    A candidate is an n-gram seen `count` times; its period is the distance between its
    first two occurrences. Walking back from the second occurrence while each token
    equals the one a period earlier finds the first token of the copy -- the point where
    the model chose to repeat rather than go on, which FTPO trains against.

    A loop copies a whole period verbatim. An enumeration -- "Second crate: base 3x4,
    height 6. Total 12." line after line -- shares long spans too, but its copy breaks at
    every item, and an alternative there changes content rather than ends a loop; such a
    candidate is passed over for the next."""
    ids = np.asarray(ids)
    after = np.nonzero(ids[start:] == close)[0]
    end = start + int(after[0]) if len(after) else len(ids)
    if end - start <= 4:  # "<think>\n\n</think>": no thought, so the whole reply
        end = len(ids)
    seen, tried = {}, set()
    for s in range(start, end - n + 1):
        gram = ids[s:s + n].tobytes()
        hits = seen.setdefault(gram, [])
        hits.append(s)
        if len(hits) < count or gram in tried:
            continue
        tried.add(gram)
        period = hits[1] - hits[0]
        at = hits[1]
        while at - 1 - period >= start and ids[at - 1] == ids[at - 1 - period]:
            at -= 1
        run = at
        while run < end and ids[run] == ids[run - period]:
            run += 1
        if run - at >= max(period, n):
            return at
    return None


def unlikelihood_loss(hidden, head, ids, negative, chunk=128, weight=None):
    """Summed -log(1 - p(next token)) over `negative` positions, each times its position
    weight when `weight` is given (zero-weight negatives drop out), like every other term.

    Rows are projected in chunks: a 248320-wide fp32 row is 1 MB, and a looping micro
    batch has hundreds of negatives -- thousands at 32K tokens of packed rollouts, which
    is why each chunk is checkpointed like the KL's (chunked_head_loss): without it every
    chunk's logits stayed alive for backward and long round 4 ran out of memory."""
    from torch.utils.checkpoint import checkpoint

    if weight is not None:
        negative = negative & (weight > 0)
    positions = negative.nonzero()
    total = hidden.new_zeros((), dtype=torch.float32)

    def penalty(state, wanted, scale):
        logits = head(state).float()
        p = (logits.gather(-1, wanted[:, None]).squeeze(-1) - logits.logsumexp(-1)).exp()
        return (-torch.log1p(-p.clamp(max=1 - 1e-6)) * scale).sum()

    for begin in range(0, len(positions), chunk):
        at = positions[begin:begin + chunk]
        state = hidden[at[:, 0], at[:, 1]]
        wanted = ids[at[:, 0], at[:, 1] + 1]
        scale = (weight[at[:, 0], at[:, 1]].float() if weight is not None
                 else torch.ones(len(at), device=state.device))
        if state.requires_grad:
            # Deterministic recompute; no RNG to restore (as chunked_head_loss).
            total = total + checkpoint(penalty, state, wanted, scale, use_reentrant=False,
                                       preserve_rng_state=False)
        else:
            total = total + penalty(state, wanted, scale)
    return total


#: Sentence openers a thinking teacher uses to second-guess itself. Capitalized forms
#: only: lowercase " actually" is ordinary prose mid-sentence.
HEDGE_OPENERS = ("Wait", "Actually", "Hmm", "Hold")


def hedge_token_ids(tokenizer):
    """Every single-token spelling of `HEDGE_OPENERS`, bare and space-prefixed."""
    found = set()
    for word in HEDGE_OPENERS:
        for form in (word, " " + word):
            pieces = tokenizer(form, add_special_tokens=False)["input_ids"]
            if len(pieces) == 1:
                found.add(int(pieces[0]))
    return np.array(sorted(found), dtype=np.int64)


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
