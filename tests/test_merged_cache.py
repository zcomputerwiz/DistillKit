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


def write(path, doc_ids, *, top_k=TOP_K, tokenizer=TOKENIZER, length=6, split="train"):
    """A capture holding `doc_ids`, with each document's tokens keyed to its id."""
    writer = OfflineCacheWriter(path, tokenizer_hash=tokenizer, anchor_layers=ANCHORS,
                                hidden_size=HIDDEN, vocab_size=VOCAB,
                                sequence_length=32, top_k=top_k, shard_tokens=64,
                                tokenizer_vocab_fingerprint=VOCAB_HASH)
    for index, doc_id in enumerate(doc_ids):
        # The token value identifies the document, so a misrouted read is visible.
        writer.append(doc_id,
                      np.full(length, index + 1, dtype=np.uint32),
                      np.zeros((length, top_k), dtype=np.uint32),
                      np.zeros((length, top_k), dtype=np.float16),
                      np.zeros((length, len(ANCHORS), HIDDEN), dtype=np.uint8),
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
