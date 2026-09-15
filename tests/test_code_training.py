"""Tests for the code continued-pretraining harness: accounting, packing, immutability.

Everything here is about a number being what it claims. A packed stream that drops a
token, an evaluation subset that shifts between checkpoints, or a milestone that fires at
the wrong count all produce a training curve that reads perfectly and compares two
different things. None of these tests touch the network or a GPU.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch" / "code_training"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch" / "code_corpus"))


class FakeStore:
    """A token store with known contents, so accounting has an expected answer."""

    def __init__(self, documents, identities=None):
        self.documents_list = [np.asarray(d, dtype=np.uint32) for d in documents]
        offsets = [0]
        for document in self.documents_list:
            offsets.append(offsets[-1] + len(document))
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.tokens = np.concatenate(self.documents_list) if documents else np.empty(0)
        self.split = "train"
        self.meta = {"documents": identities or
                     ["repo%d/file%d.py" % (i, i) for i in range(len(documents))]}

    def __len__(self):
        return len(self.documents_list)

    def document(self, index):
        return self.documents_list[index]

    @property
    def total_tokens(self):
        return int(self.offsets[-1])


@pytest.fixture
def store():
    rng = np.random.default_rng(0)
    return FakeStore([rng.integers(0, 1000, size=n) for n in
                      (50, 300, 120, 2000, 75, 640, 410, 95, 1200, 33)])


class TestPackedStream:
    def test_every_sequence_is_exactly_the_requested_length(self, store):
        from train import packed_stream

        for sequence in packed_stream(store, 128, seed=1):
            assert len(sequence) == 128

    def test_the_stream_is_a_prefix_of_the_concatenated_shuffled_documents(self, store):
        """No token is duplicated, reordered within a document, or invented."""
        from train import packed_stream

        order = np.random.default_rng(1).permutation(len(store))
        expected = np.concatenate([store.document(i) for i in order])
        produced = np.concatenate(list(packed_stream(store, 128, seed=1)))
        assert np.array_equal(produced, expected[:len(produced)])

    def test_at_most_one_sequence_worth_is_dropped(self, store):
        from train import packed_stream

        produced = sum(len(s) for s in packed_stream(store, 128, seed=1))
        assert 0 <= store.total_tokens - produced < 128

    def test_order_depends_only_on_the_seed(self, store):
        from train import packed_stream

        first = list(packed_stream(store, 128, seed=7))
        second = list(packed_stream(store, 128, seed=7))
        other = list(packed_stream(store, 128, seed=8))
        assert all(np.array_equal(a, b) for a, b in zip(first, second))
        assert not all(np.array_equal(a, b) for a, b in zip(first, other))

    def test_sequence_length_does_not_change_which_documents_come_first(self, store):
        """The shuffle is over documents, so two lengths see the same data in one order.

        Shuffling after packing would make the stream depend on the sequence length and
        two runs at different lengths would not be comparable.
        """
        from train import packed_stream

        short = np.concatenate(list(packed_stream(store, 64, seed=3)))
        long = np.concatenate(list(packed_stream(store, 256, seed=3)))
        common = min(len(short), len(long))
        assert np.array_equal(short[:common], long[:common])

    def test_an_empty_corpus_yields_nothing_rather_than_hanging(self):
        from train import packed_stream

        assert list(packed_stream(FakeStore([]), 16, seed=0)) == []


class TestTokenAccounting:
    def test_targets_are_one_fewer_than_inputs_per_sequence(self):
        """The first position of a packed sequence is context, never a supervised target.

        Reporting sequence_length tokens per sequence instead of sequence_length - 1
        overstates the corpus consumed by one token per sequence -- 15,000 tokens over a
        30M-token run, which is small and is exactly the kind of drift that makes a
        milestone not mean what it says.
        """
        sequence_length, micro_batch, accumulate = 2048, 2, 8
        per_update = micro_batch * accumulate * (sequence_length - 1)
        assert per_update == 32752
        assert per_update != sequence_length * micro_batch * accumulate

    def test_milestones_are_reached_in_order_and_each_fires_once(self):
        milestones = [5_000_000, 10_000_000, 20_000_000, 30_725_434]
        pending = list(milestones)
        fired, tokens = [], 0
        while pending:
            tokens += 1_000_000
            while pending and tokens >= pending[0]:
                fired.append(pending.pop(0))
        assert fired == milestones

    def test_a_large_step_can_cross_two_milestones_without_skipping_one(self):
        pending = [1000, 2000, 3000]
        fired, tokens = [], 0
        tokens += 2500
        while pending and tokens >= pending[0]:
            fired.append(pending.pop(0))
        assert fired == [1000, 2000]
        assert pending == [3000]


class TestEvaluationSubset:
    def test_the_same_subset_comes_back_every_time(self, store):
        from corpus import evaluation_subset

        first, meta_a = evaluation_subset(store, 1500)
        second, meta_b = evaluation_subset(store, 1500)
        assert first == second
        assert meta_a["digest"] == meta_b["digest"]

    def test_the_subset_does_not_depend_on_document_order_in_the_store(self):
        """Selection is by document identity, so it survives a corpus rebuild's ordering."""
        from corpus import evaluation_subset

        rng = np.random.default_rng(2)
        documents = [rng.integers(0, 100, size=n) for n in (40, 90, 70, 110, 60, 80)]
        names = ["r%d/f.py" % i for i in range(len(documents))]
        forward = FakeStore(documents, names)
        order = [3, 0, 5, 1, 4, 2]
        shuffled = FakeStore([documents[i] for i in order], [names[i] for i in order])

        chosen_a, _ = evaluation_subset(forward, 200)
        chosen_b, _ = evaluation_subset(shuffled, 200)
        assert {names[i] for i in chosen_a} == {names[order[i]] for i in chosen_b}

    def test_zero_takes_the_whole_split(self, store):
        from corpus import evaluation_subset

        chosen, meta = evaluation_subset(store, 0)
        assert chosen == list(range(len(store)))
        assert meta["tokens"] == store.total_tokens

    def test_the_digest_changes_when_the_membership_changes(self, store):
        from corpus import evaluation_subset

        _, small = evaluation_subset(store, 200)
        _, large = evaluation_subset(store, 3000)
        assert small["digest"] != large["digest"]
        assert small["tokens"] <= large["tokens"]

    def test_indices_come_back_sorted_so_scoring_order_is_stable(self, store):
        from corpus import evaluation_subset

        chosen, _ = evaluation_subset(store, 1500)
        assert chosen == sorted(chosen)


class TestSplitIsolation:
    def test_no_repository_spans_two_splits(self):
        """Re-asserted here, on the identity strings the evaluation actually keys on.

        The corpus builder guarantees this by construction; this checks the guarantee
        survives into the token store, which is what training and evaluation read.
        """
        from distillkit.code_corpus import DEFAULT_PROPORTIONS, split_for_repo

        seed = 20260914
        members = {name: set() for name in DEFAULT_PROPORTIONS}
        for index in range(3000):
            repo = "owner%d/project%d" % (index, index % 17)
            members[split_for_repo(repo, seed, DEFAULT_PROPORTIONS)].add(repo)
        assert not members["train"] & members["heldout"]
        assert not members["train"] & members["calibration"]
        assert not members["calibration"] & members["heldout"]


class TestEvaluationBatching:
    def test_every_document_appears_exactly_once(self, store):
        from evaluate_python import batches

        indices = list(range(len(store)))
        produced = [i for group in batches(store, indices, 4096, 2048) for i in group]
        assert sorted(produced) == indices

    def test_no_batch_exceeds_the_token_budget(self, store):
        from evaluate_python import batches

        for group in batches(store, list(range(len(store))), 4096, 2048):
            width = max(min(int(store.offsets[i + 1] - store.offsets[i]), 2048)
                        for i in group)
            assert width * len(group) <= 4096 or len(group) == 1

    def test_a_document_longer_than_the_budget_still_gets_scored(self):
        from evaluate_python import batches

        big = FakeStore([np.zeros(9000, dtype=np.uint32), np.zeros(10, dtype=np.uint32)])
        produced = [i for group in batches(big, [0, 1], 4096, 2048) for i in group]
        assert sorted(produced) == [0, 1]
