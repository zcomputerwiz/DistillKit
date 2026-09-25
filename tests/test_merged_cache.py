"""Two captures read as one corpus, and the ways that must not silently succeed.

Capturing 5M tokens from the 27B teacher costs hours, so a corpus that grows should be
an extra capture rather than a recapture of what is already on disk. `MergedCache` puts
several captures behind one cache's interface for exactly that.

What it must refuse is the interesting half. A cache carries the assumptions the
grouped-tail objective is derived under -- which tokenizer the ids belong to, how wide
the kept head is, what temperature it was generated at -- and two captures that disagree
about any of them produce a wrong loss rather than an error. So the mismatch cases are
tested alongside the working one.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch" / "dense_gr"))

from distillkit.offline_cache import OfflineCacheWriter  # noqa: E402
from teacher_kl import MergedCache  # noqa: E402

VOCAB, TOP_K, ANCHORS, HIDDEN = 128, 4, [0, 1], 8
TOKENIZER = "a" * 64
VOCAB_HASH = "b" * 64


def write(path, doc_ids, *, top_k=TOP_K, tokenizer=TOKENIZER, length=6, split="train",
          tokens=None):
    """A capture holding `doc_ids`, with each document's tokens keyed to its id."""
    writer = OfflineCacheWriter(path, tokenizer_hash=tokenizer, anchor_layers=ANCHORS,
                                hidden_size=HIDDEN, vocab_size=VOCAB,
                                sequence_length=32, top_k=top_k, shard_tokens=64,
                                tokenizer_vocab_fingerprint=VOCAB_HASH)
    for index, doc_id in enumerate(doc_ids):
        # The token value identifies the document, so a misrouted read is visible.
        ids = (np.asarray(tokens[doc_id], dtype=np.uint32) if tokens
               else np.full(length, index + 1, dtype=np.uint32))
        width = len(ids)
        writer.append(doc_id, ids,
                      np.zeros((width, top_k), dtype=np.uint32),
                      np.zeros((width, top_k), dtype=np.float16),
                      np.zeros((width, len(ANCHORS), HIDDEN), dtype=np.uint8),
                      split=split)
    writer.close()
    return path


def test_merged_cache_reads_both_captures(tmp_path):
    merged = MergedCache([write(tmp_path / "one", ["a", "b"]),
                          write(tmp_path / "two", ["c"])])
    assert sorted(merged.documents) == ["a", "b", "c"]
    assert sorted(merged.document_ids("train")) == ["a", "b", "c"]
    assert merged.document_ids("eval") == []
    assert merged.manifest["top_k"] == TOP_K
    merged.close()


def test_merged_cache_routes_reads_to_the_owning_capture(tmp_path):
    merged = MergedCache([write(tmp_path / "one", ["a", "b"]),
                          write(tmp_path / "two", ["c"])])
    # "a" and "c" are both the first document of their own capture, so a merge that
    # routed by position rather than by owner would hand back the same tokens twice.
    first = merged.read_document("a", include_hidden_states=False)["input_ids"]
    second = merged.read_document("c", include_hidden_states=False)["input_ids"]
    assert set(first.tolist()) == {1} and set(second.tolist()) == {1}
    assert merged.read_document("b", include_hidden_states=False)["input_ids"].tolist() \
        == [2] * 6
    merged.close()


def test_merged_cache_refuses_a_document_in_both_captures(tmp_path):
    with pytest.raises(ValueError, match="in both"):
        MergedCache([write(tmp_path / "one", ["a", "b"]),
                     write(tmp_path / "two", ["b"])])


def test_merged_cache_refuses_a_different_top_k(tmp_path):
    with pytest.raises(ValueError, match="top_k"):
        MergedCache([write(tmp_path / "one", ["a"]),
                     write(tmp_path / "two", ["b"], top_k=8)])


def test_merged_cache_refuses_a_different_tokenizer(tmp_path):
    with pytest.raises(ValueError, match="tokenizer_hash"):
        MergedCache([write(tmp_path / "one", ["a"]),
                     write(tmp_path / "two", ["b"], tokenizer="c" * 64)])


MARKER = [90, 91]


def test_first_response_finds_the_marker_run():
    from teacher_kl import first_response

    assert first_response([5, 90, 91, 7, 8], MARKER) == 3
    # The first marker, not the last: a multi-turn document has several, and what the
    # cap decides is whether any response survives it.
    assert first_response([90, 91, 7, 90, 91, 8], MARKER) == 2
    assert first_response([5, 6, 7], MARKER) is None
    # A marker split across the cap boundary is not a marker.
    assert first_response([5, 6, 90], MARKER) is None


def test_min_answer_tokens_drops_documents_that_are_all_prompt(tmp_path):
    from teacher_kl import CachedTeacher

    # The answer starts at index 8 in the first two. The last position of a document
    # carries no ground truth and is not scored, so a length-16 document has 7 scored
    # answer positions (8..14) and a length-11 one has 2.
    tokens = {"answer": [5] * 6 + MARKER + [7] * 8,
              "short": [5] * 6 + MARKER + [7] * 3,
              "prompt": [5] * 10}
    path = write(tmp_path / "one", list(tokens), tokens=tokens)
    assert len(CachedTeacher(path, "train", device="cpu")) == 3

    kept = CachedTeacher(path, "train", device="cpu", answer_marker=MARKER,
                         min_answer_tokens=1)
    assert sorted(kept.ids) == ["answer", "short"]
    assert kept.dropped_all_prompt == 1

    strict = CachedTeacher(path, "train", device="cpu", answer_marker=MARKER,
                           min_answer_tokens=4)
    assert strict.ids == ["answer"]
    assert strict.dropped_all_prompt == 2
    # Reported tokens follow the documents that survived, or the budget is wrong.
    assert strict.tokens == 16


def test_min_answer_tokens_counts_only_what_the_cap_keeps(tmp_path):
    from teacher_kl import CachedTeacher

    # The answer starts at index 12, so a cap of 14 leaves one scored answer position
    # and a cap of 13 leaves none -- this is the case the whole option exists for.
    tokens = {"late": [5] * 10 + MARKER + [7] * 8}
    path = write(tmp_path / "one", list(tokens), tokens=tokens)
    assert CachedTeacher(path, "train", device="cpu", max_length=14,
                         answer_marker=MARKER, min_answer_tokens=1).ids == ["late"]
    assert CachedTeacher(path, "train", device="cpu", max_length=13,
                         answer_marker=MARKER, min_answer_tokens=1).ids == []


def test_min_answer_tokens_needs_a_marker(tmp_path):
    from teacher_kl import CachedTeacher

    path = write(tmp_path / "one", ["a"])
    with pytest.raises(ValueError, match="answer_marker"):
        CachedTeacher(path, "train", device="cpu", min_answer_tokens=1)


def test_cached_teacher_takes_one_path_or_several(tmp_path):
    from teacher_kl import CachedTeacher
    from distillkit.offline_cache import OfflineTeacherCache

    one = write(tmp_path / "one", ["a", "b"])
    two = write(tmp_path / "two", ["c"])
    single = CachedTeacher(one, "train", device="cpu")
    assert isinstance(single.cache, OfflineTeacherCache)
    assert len(single) == 2
    both = CachedTeacher([one, two], "train", device="cpu")
    assert isinstance(both.cache, MergedCache)
    assert len(both) == 3
    assert both.tokens == single.tokens + 6


def _doc(prompt, answer):
    """A document whose answer starts right after `prompt` tokens."""
    return [5] * prompt + MARKER + [7] * answer


def test_grouping_keeps_the_remainder_and_is_batch_size_independent(tmp_path):
    """The last, partial group is a smaller batch rather than a discarded one.

    `break` on a short group threw away up to `rows - 1` documents every pass, and
    because the group boundaries move with the batch size, one card and two cards were
    not training on the same examples. Whatever the batch size, every document that
    survives the width floor has to appear exactly once.
    """
    from teacher_kl import CachedTeacher

    tokens = {"d%02d" % i: [5] * (16 + i) for i in range(11)}
    path = write(tmp_path / "one", list(tokens), tokens=tokens)
    teacher = CachedTeacher(path, "train", device="cpu")

    seen = {}
    for size in (1, 2, 3, 4):
        members = [doc for group, _ in teacher._groups(size) for doc in group]
        assert len(members) == len(set(members)), "a document appears twice at size %d" % size
        seen[size] = set(members)
    assert seen[1] == seen[2] == seen[3] == seen[4], (
        "changing the batch size changed which documents are trained on")
    assert seen[1] == set(tokens)


def test_grouping_cannot_truncate_an_answer_the_filter_admitted(tmp_path):
    """The answer check belongs at the retained width, not at the cap.

    A group is cut to its shortest member and floored to the routing block, so a
    document can pass the check at its own length and be scored at a shorter one. The
    filter has to run at the length the document is actually trained at.
    """
    from teacher_kl import CachedTeacher

    # "late" carries its answer from index 22; grouped with "early" it is cut to 20
    # and the answer is gone, while on its own it would keep two scored answer tokens.
    tokens = {"early": _doc(prompt=2, answer=16), "late": _doc(prompt=20, answer=8)}
    path = write(tmp_path / "one", list(tokens), tokens=tokens)
    teacher = CachedTeacher(path, "train", device="cpu",
                            answer_marker=MARKER, min_answer_tokens=2)
    assert sorted(teacher.ids) == ["early", "late"], "both pass at their own length"

    groups = teacher._groups(2)
    survivors = {doc for group, _ in groups for doc in group}
    assert survivors == {"early", "late"}
    assert teacher.dropped_truncated_answer == 0

    for group, width in groups:
        for doc_id in group:
            assert teacher._kept_answer(doc_id, width) >= 2


def test_planned_tokens_counts_what_grouping_keeps(tmp_path):
    """`--passes` sized off the cache overstates the budget by whatever grouping trims."""
    from teacher_kl import CachedTeacher

    tokens = {"d%02d" % i: [5] * (16 + i) for i in range(8)}
    path = write(tmp_path / "one", list(tokens), tokens=tokens)
    teacher = CachedTeacher(path, "train", device="cpu")

    planned = teacher.planned_tokens(4)
    assert planned == sum(len(g) * (w - 1) for g, w in teacher._groups(4))
    assert planned == teacher.tokens - len(tokens)


@pytest.mark.parametrize("minimum", [0, 2])
def test_prefixes_filtering_and_targets_are_independent_of_batching(tmp_path, minimum):
    from teacher_kl import CachedTeacher

    tokens = {"a": _doc(2, 11), "b": _doc(9, 5), "c": _doc(10, 6),
              "d": _doc(1, 5), "e": _doc(15, 8)}
    teacher = CachedTeacher(write(tmp_path / "cache", list(tokens), tokens=tokens),
                            device="cpu", answer_marker=MARKER, min_answer_tokens=minimum)
    reference = None
    for rows, budget in ((1, None), (2, None), (6, None), (1, 48)):
        groups = teacher._groups(rows, 8, budget)
        prefixes = {doc: width for group, width in groups for doc in group}
        assert len(prefixes) == sum(len(group) for group, _ in groups)
        reference = prefixes if reference is None else reference
        assert prefixes == reference
        assert teacher.planned_tokens(rows, 8, budget) == sum(w - 1 for w in reference.values())
        assert teacher.shapes(rows, 8, budget) == sorted({(len(g), w) for g, w in groups})
        for group, width in groups:
            if minimum:
                assert all(teacher._kept_answer(d, width) >= minimum for d in group)
    teacher.close()


def test_the_held_out_sample_covers_every_capture(tmp_path):
    """Taking the first N of a merged corpus takes them all from the first capture."""
    from teacher_kl import CachedTeacher

    first = {"a%02d" % i: [5] * 8 for i in range(20)}
    second = {"b%02d" % i: [6] * 8 for i in range(20)}
    one = write(tmp_path / "one", list(first), tokens=first, split="eval")
    two = write(tmp_path / "two", list(second), tokens=second, split="eval")

    teacher = CachedTeacher([one, two], "eval", device="cpu")
    assert len({source for source, _ in teacher.stratified(10)}) == 2, (
        "the sample came from a single capture")

    # Naming the captures the other way round selects the same documents, or a
    # held-out series is not comparable between runs that ordered them differently.
    flipped = CachedTeacher([two, one], "eval", device="cpu")
    assert ({doc for _, doc in teacher.stratified(10)}
            == {doc for _, doc in flipped.stratified(10)})


def test_stratified_sample_fills_rounding_shortfall():
    from teacher_kl import CachedTeacher

    teacher = CachedTeacher.__new__(CachedTeacher)
    teacher.sources = lambda: {s: [s + str(i) for i in range(4)] for s in ("a", "b", "c")}
    picked = teacher.stratified(4)
    assert len(picked) == len({doc for _, doc in picked}) == 4
    assert {source for source, _ in picked} == {"a", "b", "c"}
    assert len(teacher.stratified(100)) == 12


def test_splitting_a_step_into_microbatches_does_not_change_the_gradient(tmp_path):
    """The acceptance test for the reduction: same examples, different boundaries.

    Every term is a mean over its own scored positions, so dividing each micro-batch by
    `accumulate` averages means of different denominators. Documents here run 134 to
    1024 tokens, so with one document per micro-batch a short document carried the same
    weight as a document eight times its length, and the gradient moved with where the
    accumulation boundaries fell. Weighting by each micro-batch's share of the step's
    scored tokens makes an accumulated step equal to one forward over the same rows.
    """
    import torch
    from torch import nn
    from torch.nn import functional as F

    from teacher_kl import accumulation_shares

    torch.manual_seed(0)
    hidden_size, vocab = 8, 16
    head = nn.Linear(hidden_size, vocab, bias=False)
    # Deliberately uneven: 2 rows of 12, 1 row of 5, 3 rows of 7.
    chunks = [torch.randint(0, vocab, shape) for shape in ((2, 12), (1, 5), (3, 7))]
    states = [torch.randn(*chunk.shape, hidden_size) for chunk in chunks]

    def scored(state, chunk):
        logits = head(state)[:, :-1].reshape(-1, vocab)
        return logits, chunk[:, 1:].reshape(-1)

    # One forward over every scored position, which is what the step should equal.
    logits = torch.cat([scored(s, c)[0] for s, c in zip(states, chunks)])
    targets = torch.cat([scored(s, c)[1] for s, c in zip(states, chunks)])
    F.cross_entropy(logits, targets).backward()
    reference = head.weight.grad.detach().clone()
    head.zero_grad(set_to_none=True)

    counts, total = accumulation_shares(chunks)
    assert counts == [22, 4, 18] and total == 44
    for state, chunk, count in zip(states, chunks, counts):
        piece_logits, piece_targets = scored(state, chunk)
        (F.cross_entropy(piece_logits, piece_targets) * (count / total)).backward()
    torch.testing.assert_close(head.weight.grad, reference, rtol=1e-5, atol=1e-7)
    head.zero_grad(set_to_none=True)

    # And the rule it replaces does not have the property, or this test proves nothing.
    for state, chunk in zip(states, chunks):
        piece_logits, piece_targets = scored(state, chunk)
        (F.cross_entropy(piece_logits, piece_targets) / len(chunks)).backward()
    assert not torch.allclose(head.weight.grad, reference, rtol=1e-3, atol=1e-5)


def test_excluded_documents_are_gone_before_anything_is_planned(tmp_path):
    """Contaminated documents must not reach the plan, the sample or the budget."""
    from teacher_kl import CachedTeacher

    tokens = {"d%02d" % i: [5] * (16 + i) for i in range(6)}
    path = write(tmp_path / "one", list(tokens), tokens=tokens)
    full = CachedTeacher(path, "train", device="cpu")
    kept = CachedTeacher(path, "train", device="cpu", exclude={"d01", "d04"})
    assert kept.excluded == 2
    assert sorted(kept.ids) == ["d00", "d02", "d03", "d05"]
    assert kept.tokens == full.tokens - (16 + 1) - (16 + 4)
    assert not {"d01", "d04"} & {d for group, _ in kept._groups(2) for d in group}


def test_an_exclusion_list_for_another_corpus_is_refused(tmp_path):
    """Ids that exist in no capture mean the wrong list, which would exclude nothing."""
    from teacher_kl import CachedTeacher

    path = write(tmp_path / "one", ["a", "b"])
    with pytest.raises(ValueError, match="in none of these captures"):
        CachedTeacher(path, "train", device="cpu", exclude={"a", "not-here"})
