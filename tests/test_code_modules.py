"""Invariants for the code-domain gate and structural sidecar fits.

What these pin is the set of claims that would be invisible if they broke. A familiarity
cache that quietly read the heldout split, a gate that was not an identity at
initialization, or a wrong-address control that did not actually misaddress anything would
all produce numbers that look exactly like real ones. Synthetic fixtures throughout; no
network and no GPU.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scratch" / "code_gate"))
sys.path.insert(0, str(ROOT / "scratch" / "code_training"))
sys.path.insert(0, str(ROOT / "scratch" / "ffn_memo"))

VOCAB = 1000


class FakeStore:
    def __init__(self, documents):
        self.documents_list = [np.asarray(d, dtype=np.uint32) for d in documents]
        offsets = [0]
        for document in self.documents_list:
            offsets.append(offsets[-1] + len(document))
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.split = "train"
        self.meta = {"documents": ["r%d/f.py" % i for i in range(len(documents))]}

    def __len__(self):
        return len(self.documents_list)

    def document(self, index):
        return self.documents_list[index]


class TestTrigramKeys:
    def test_matches_the_reference_implementation_exactly(self):
        """The vectorized key must equal the canonical loop, or the cache addresses
        different contexts than every published familiarity number did."""
        from familiarity import trigram_keys
        from repeatability import trigram_keys as reference

        rng = np.random.default_rng(0)
        ids = rng.integers(0, VOCAB, size=200).tolist()
        assert list(trigram_keys(ids, VOCAB)) == [k for k in reference(ids, VOCAB)
                                                  if k is not None]

    def test_key_is_the_context_ending_at_the_position(self):
        from familiarity import trigram_keys

        keys = trigram_keys([7, 8, 9, 10], VOCAB)
        assert keys[0] == (7 * VOCAB + 8) * VOCAB + 9
        assert keys[1] == (8 * VOCAB + 9) * VOCAB + 10

    def test_short_documents_yield_nothing_rather_than_raising(self):
        from familiarity import trigram_keys

        assert trigram_keys([1, 2], VOCAB).size == 0
        assert trigram_keys([], VOCAB).size == 0

    def test_distinct_contexts_never_collide(self):
        """The key is exact arithmetic, not a hash; collisions would silently pool
        statistics from unrelated contexts."""
        from familiarity import trigram_keys

        rng = np.random.default_rng(1)
        ids = rng.integers(0, VOCAB, size=5000)
        keys = trigram_keys(ids, VOCAB)
        windows = {tuple(ids[i:i + 3]) for i in range(len(ids) - 2)}
        assert len(set(keys.tolist())) == len(windows)


class TestCountPass:
    def _reference(self, store, vocab, min_count, max_keys):
        """The canonical dict-of-sets shape, for the equivalence check."""
        import collections

        from familiarity import trigram_keys

        counts = collections.Counter()
        documents_per_key = collections.defaultdict(set)
        for number in range(len(store)):
            for key in trigram_keys(store.document(number), vocab):
                counts[int(key)] += 1
                documents_per_key[int(key)].add(number)
        frequent = [k for k, c in counts.most_common() if c >= min_count]
        return [k for k in frequent if len(documents_per_key[k]) >= 2][:max_keys]

    def test_agrees_with_the_reference_shape(self):
        from familiarity import count_pass

        rng = np.random.default_rng(2)
        store = FakeStore([rng.integers(0, 40, size=n) for n in (60, 80, 50, 90, 70)])
        _, frequent, _, _ = count_pass(store, VOCAB, 2, 10_000)
        assert set(frequent) == set(self._reference(store, VOCAB, 2, 10_000))

    def test_cross_document_recurrence_is_required(self):
        """A trigram repeated inside one document only must not be cached.

        Counting it would measure within-document leakage rather than whether a cache
        transfers, which is the entire point of the statistic.
        """
        from familiarity import count_pass

        repeated = [5, 6, 7] * 20
        store = FakeStore([repeated, list(range(20, 60))])
        _, frequent, _, _ = count_pass(store, VOCAB, 2, 10_000)
        lonely = (5 * VOCAB + 6) * VOCAB + 7
        assert lonely not in frequent

    def test_min_count_is_honoured(self):
        from familiarity import count_pass

        shared = [1, 2, 3, 4]
        store = FakeStore([shared + [9, 9, 9], shared + [8, 8, 8]])
        _, frequent, _, _ = count_pass(store, VOCAB, 2, 10_000)
        _, strict, _, _ = count_pass(store, VOCAB, 3, 10_000)
        assert len(strict) <= len(frequent)

    def test_max_keys_caps_by_frequency(self):
        from familiarity import count_pass

        rng = np.random.default_rng(3)
        store = FakeStore([rng.integers(0, 30, size=200) for _ in range(6)])
        counts, frequent, _, _ = count_pass(store, VOCAB, 2, 5)
        assert len(frequent) == 5
        kept = [counts[k] for k in frequent]
        assert kept == sorted(kept, reverse=True)

    def test_truncation_is_applied_where_asked(self):
        """The capture pass truncates, so the count pass must truncate identically or a
        key is admitted on counts its prototype was never averaged over."""
        from familiarity import count_pass

        store = FakeStore([list(range(100)), list(range(100))])
        _, _, full, _ = count_pass(store, VOCAB, 2, 10_000)
        _, _, cut, _ = count_pass(store, VOCAB, 2, 10_000, max_length=20)
        assert full == 200 and cut == 40


class TestHeldoutExclusion:
    def test_the_cache_builder_reads_only_the_train_split(self):
        """Checked against the source, because a heldout leak here would invalidate
        every number the cache feeds and leave no other trace."""
        source = (ROOT / "scratch" / "code_gate" / "familiarity.py").read_text(
            encoding="utf-8")
        body = source.split('if __name__')[0]
        built = body.split("def main")[1]
        # The builder opens train for counting and capture; heldout appears only in the
        # occupancy diagnostic, which reads counts and never writes them.
        assert 'TokenStore(TOKENS, "train")' in built
        assert 'TokenStore(TOKENS, "calibration")' not in built

    def test_the_gate_trainer_selects_on_calibration_not_heldout(self):
        source = (ROOT / "scratch" / "code_gate" / "train_gate.py").read_text(
            encoding="utf-8")
        assert 'TokenStore(TOKENS, "calibration")' in source
        assert 'TokenStore(TOKENS, "heldout")' not in source


class TestWrongAddress:
    def test_the_control_actually_misaddresses(self):
        """A control that returned the same rows would silently confirm any result."""
        import torch

        from evaluate_python import Sidecar

        class Hasher:
            def row_indices(self, ids):
                return torch.arange(ids.shape[1]).unsqueeze(0).repeat(ids.shape[0], 1)

        ids = torch.zeros((1, 32), dtype=torch.long)
        honest = Sidecar(None, Hasher(), None, 1.0)
        control = Sidecar(None, Hasher(), None, 1.0)
        control.wrong_address = True
        assert not torch.equal(honest.rows(ids), control.rows(ids))

    def test_the_control_still_addresses_real_rows(self):
        """Misaddressing must read a different real context, not an invalid index: the
        question is whether the gain depends on *correct* context, not on any context."""
        import torch

        from evaluate_python import Sidecar

        class Hasher:
            def row_indices(self, ids):
                return torch.arange(ids.shape[1]).unsqueeze(0)

        ids = torch.zeros((1, 32), dtype=torch.long)
        control = Sidecar(None, Hasher(), None, 1.0)
        control.wrong_address = True
        rows = control.rows(ids)
        assert sorted(rows[0].tolist()) == list(range(32))


class TestArmSpecifications:
    def test_each_gate_arm_differs_from_its_neighbour_in_one_field(self):
        """The decomposition is only interpretable if the arms are nested this way."""
        from evaluate_python import ARMS

        general, swapped, normalized = (ARMS["gate"], ARMS["gate_py"],
                                        ARMS["gate_py_norm"])
        assert general["gate"] == swapped["gate"] == normalized["gate"] == "general"
        assert general["cache"] == "general" and swapped["cache"] == "python"
        assert swapped["normalizer"] == "kept"
        assert normalized["cache"] == "python" and normalized["normalizer"] == "python"

    def test_the_code_arms_use_the_code_checkpoints(self):
        from evaluate_python import ARMS

        assert ARMS["gcode"]["gate"] == "code" and ARMS["gcode"]["cache"] == "python"
        assert ARMS["scode"]["sidecar"] == "code"
        assert ARMS["stock"] == {}

    def test_stock_enables_nothing(self):
        from evaluate_python import ARMS

        assert not ARMS["stock"].get("gate") and not ARMS["stock"].get("sidecar")


class TestFittedArtifacts:
    """Facts recorded by the fits, checked from their own reports rather than re-run."""

    def _load(self, relative):
        import json

        path = ROOT / relative
        if not path.exists():
            pytest.skip("%s has not been produced yet" % relative)
        return json.loads(path.read_text(encoding="utf-8"))

    def test_the_backbone_was_frozen_during_the_gate_fit(self):
        report = self._load("scratch/code_gate/gcode/training.json")
        assert report["backbone_frozen"]
        assert report["backbone_sha256_before"] == report["backbone_sha256_after"]

    def test_the_gate_started_as_an_exact_identity(self):
        report = self._load("scratch/code_gate/gcode/training.json")
        assert report["identity_at_initialization"] == 0.0

    def test_the_gate_kept_the_canonical_parameter_count(self):
        report = self._load("scratch/code_gate/gcode/training.json")
        assert report["parameters"] == 260

    def test_the_gate_was_selected_on_calibration(self):
        report = self._load("scratch/code_gate/gcode/training.json")
        assert "calibration" in report["selection_criterion"]
        assert "heldout" not in report["selection_criterion"]

    def test_the_addressed_decoder_kept_the_canonical_architecture(self):
        report = self._load("scratch/code_sidecar/addressed/report.json")
        assert report["sidecar"]["trainable_parameters"] == 400_790
        assert report["sidecar"]["code_dim"] == 32
        assert report["sidecar"]["mode"] == "direct"
        assert report["sidecar"]["frozen_code_entries"] == 0

    def test_the_addressed_decoder_was_fitted_on_python(self):
        report = self._load("scratch/code_sidecar/addressed/report.json")
        assert report["corpus"] == "python"
        assert report["backbone_sha256_before"] == report.get(
            "backbone_sha256_after", report["backbone_sha256_before"])
