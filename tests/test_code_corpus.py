"""Tests for the deterministic Stack v2 corpus policy.

Everything here runs on synthetic fixtures. The network path is exercised by
``scratch/code_corpus/probe.py``, which is an integration probe rather than a unit
test; what these tests pin is the part that has to be right *silently* -- a split that
drifts, a checksum that is not actually checked, or a duplicate that is counted twice
produces a corpus that looks fine and invalidates the experiment it feeds.
"""

import gzip
import hashlib
import sys
import zlib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch" / "code_corpus"))

from distillkit.code_corpus import (
    DEFAULT_PROPORTIONS, BenchmarkIndex, FilterLimits, SplitPolicy, blob_url,
    decode_source, decompress_blob, filter_reason, normalize_code, shingles,
    split_for_repo, token_filter_reason, verify_blob)

REPOS = ["owner%d/project%d" % (i, i * 7 % 13) for i in range(4000)]


def metadata(**overrides):
    record = {"is_generated": False, "is_vendor": False, "length_bytes": 4096,
              "license_type": "permissive"}
    record.update(overrides)
    return record


class TestSplitDeterminism:
    def test_repeated_calls_agree(self):
        first = [split_for_repo(name) for name in REPOS]
        second = [split_for_repo(name) for name in reversed(REPOS)][::-1]
        assert first == second

    def test_independent_of_seen_order_and_corpus_contents(self):
        """The assignment is a pure function of the name, which is the whole point.

        A builder that filled ``train`` first and swept the remainder into ``heldout``
        would let shard ordering decide the holdout, and shard ordering in The Stack v2
        correlates with crawl date.
        """
        import random

        shuffled = list(REPOS)
        random.Random(0).shuffle(shuffled)
        assert {name: split_for_repo(name) for name in shuffled} == \
               {name: split_for_repo(name) for name in REPOS}

    def test_seed_changes_assignment(self):
        changed = sum(split_for_repo(n) != split_for_repo(n, seed=99) for n in REPOS)
        assert changed > len(REPOS) // 8

    def test_proportions_are_approximately_honoured(self):
        import collections

        counts = collections.Counter(split_for_repo(name) for name in REPOS)
        for name, share in DEFAULT_PROPORTIONS.items():
            assert counts[name] / len(REPOS) == pytest.approx(share, abs=0.02)

    def test_every_repo_lands_in_exactly_one_split(self):
        members = {name: set() for name in DEFAULT_PROPORTIONS}
        for repo in REPOS:
            members[split_for_repo(repo)].add(repo)
        assert sum(len(v) for v in members.values()) == len(REPOS)
        assert not members["train"] & members["heldout"]
        assert not members["train"] & members["calibration"]
        assert not members["calibration"] & members["heldout"]

    def test_all_files_of_a_repo_share_a_split(self):
        """Repository-level isolation follows from hashing the repository, not the file."""
        repo = "numpy/numpy"
        paths = ["/core/%d.py" % i for i in range(50)]
        assert len({split_for_repo(repo) for _ in paths}) == 1


class TestBlobRetrievalPolicy:
    def test_url_requires_a_real_identifier(self):
        assert blob_url("a" * 40).endswith("a" * 40)
        for bad in ("", "xyz", "A" * 40, "a" * 39, "../etc/passwd"):
            with pytest.raises(ValueError):
                blob_url(bad)

    def test_gzip_and_bare_deflate_both_decompress(self):
        payload = b"import os\n" * 100
        assert decompress_blob(gzip.compress(payload)) == payload
        assert decompress_blob(zlib.compress(payload)) == payload

    def test_corrupt_blob_raises_rather_than_returning_garbage(self):
        with pytest.raises(Exception):
            decompress_blob(b"not compressed at all")

    def test_checksum_accepts_only_the_addressing_content(self):
        payload = b"def main():\n    return 0\n"
        digest = hashlib.sha1(payload).hexdigest()
        assert verify_blob(payload, digest)
        assert not verify_blob(payload + b"\n", digest)
        assert not verify_blob(payload, "0" * 40)

    def test_decode_prefers_declared_encoding(self):
        assert decode_source("é = 1".encode("utf-8"), "UTF-8") == "é = 1"
        assert decode_source("é = 1".encode("latin-1"), "ISO-8859-1") == "é = 1"

    def test_undecodable_and_binary_return_none_rather_than_replacement_text(self):
        assert decode_source(b"\xff\xfe\x00\x01 bad", "UTF-8") is None
        assert decode_source(b"print(1)\x00\x00", "UTF-8") is None
        assert decode_source(b"\x80\x81\x82", "utf-8") is None

    def test_unknown_encoding_falls_back_to_utf8(self):
        assert decode_source(b"x = 1", "not-a-codec") == "x = 1"


class TestFilters:
    def test_metadata_flags_are_honoured(self):
        limits = FilterLimits()
        assert filter_reason(metadata(is_generated=True), limits) == "generated"
        assert filter_reason(metadata(is_vendor=True), limits) == "vendor"
        assert filter_reason(metadata(), limits) is None

    def test_sizes(self):
        limits = FilterLimits(min_bytes=64, max_bytes=1000)
        assert filter_reason(metadata(length_bytes=10), limits) == "empty"
        assert filter_reason(metadata(length_bytes=0), limits) == "empty"
        assert filter_reason(metadata(length_bytes=5000), limits) == "oversized"
        assert filter_reason(metadata(length_bytes=500), limits) is None

    def test_reasons_are_exclusive_so_counters_partition(self):
        limits = FilterLimits()
        record = metadata(is_vendor=True, is_generated=True, length_bytes=0)
        assert filter_reason(record, limits) == "generated"

    def test_token_oversize_is_caught_where_bytes_cannot_see_it(self):
        """The regression: one retrieved file was 25% of an entire training split.

        1,035,623 bytes tokenizing to 1,035,618 tokens -- obfuscated data, one token per
        byte -- sat comfortably under a 1 MiB byte cap. Only the token count reveals it.
        """
        limits = FilterLimits()
        assert token_filter_reason(1_035_618, 1_035_623, limits) == "token_oversized"
        assert token_filter_reason(1400, 5000, limits) is None

    def test_token_density_catches_obfuscated_content(self):
        limits = FilterLimits(max_tokens=10 ** 9, min_bytes_per_token=1.8)
        assert token_filter_reason(5000, 5100, limits) == "token_dense"
        assert token_filter_reason(5000, 20000, limits) is None
        # machine-generated but genuine Python sits around two bytes per token
        assert token_filter_reason(5000, 10_500, limits) is None

    def test_zero_tokens_is_empty_not_a_division_error(self):
        assert token_filter_reason(0, 500, FilterLimits()) == "empty"

    def test_license_filter_is_opt_in(self):
        record = metadata(license_type="no_license")
        assert filter_reason(record, FilterLimits()) is None
        assert filter_reason(
            record, FilterLimits(license_types=frozenset({"permissive"}))) == "license"


class TestSplitPolicy:
    def test_admits_until_target_then_only_known_repos(self):
        policy = SplitPolicy(targets={"train": 1000}, overshoot=1.5)
        assert policy.admits("train", "a/b")
        policy.record("train", "a/b", 1000)
        # at target: existing repositories finish, new ones are turned away
        assert policy.admits("train", "a/b")
        assert not policy.admits("train", "c/d")

    def test_hard_close_at_overshoot(self):
        policy = SplitPolicy(targets={"train": 1000}, overshoot=1.1)
        policy.record("train", "a/b", 1200)
        assert not policy.admits("train", "a/b")

    def test_unknown_split_is_never_admitted(self):
        policy = SplitPolicy(targets={"train": 10})
        assert not policy.admits("heldout", "a/b")

    def test_complete_requires_every_split(self):
        policy = SplitPolicy(targets={"train": 10, "heldout": 10})
        policy.record("train", "a/b", 10)
        assert not policy.complete
        policy.record("heldout", "c/d", 10)
        assert policy.complete

    def test_token_accounting_is_exact(self):
        policy = SplitPolicy(targets={"train": 10_000})
        for index in range(100):
            policy.record("train", "repo%d" % (index % 7), 37)
        assert policy.tokens["train"] == 3700
        assert len(policy.repos["train"]) == 7


SOLUTION = (
    "def longest_common_subsequence(first, second):\n"
    "    rows = len(first) + 1\n"
    "    cols = len(second) + 1\n"
    "    table = [[0] * cols for _ in range(rows)]\n"
    "    for i in range(1, rows):\n"
    "        for j in range(1, cols):\n"
    "            if first[i - 1] == second[j - 1]:\n"
    "                table[i][j] = table[i - 1][j - 1] + 1\n"
    "            else:\n"
    "                table[i][j] = max(table[i - 1][j], table[i][j - 1])\n"
    "    return table[rows - 1][cols - 1]\n")

ORDINARY = (
    "import logging\n\n"
    "LOG = logging.getLogger(__name__)\n\n"
    "class InvoiceExporter:\n"
    "    def __init__(self, session, currency='USD'):\n"
    "        self.session = session\n"
    "        self.currency = currency\n\n"
    "    def export(self, start, end):\n"
    "        rows = self.session.query(Invoice).filter(Invoice.issued.between(start, end))\n"
    "        LOG.info('exporting %d invoices', rows.count())\n"
    "        return [row.as_dict() for row in rows]\n")

#: Generic dynamic-programming and literal-list content: present in benchmark
#: solutions and in thousands of unrelated files. The first matcher flagged files on
#: exactly this, which is what the identifier floor and per-problem scoring exist to
#: prevent.
GENERIC = (
    "values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\n"
    "grid = [[0] * 12 for _ in range(12)]\n"
    "for i in range(1, 12):\n"
    "    for j in range(1, 12):\n"
    "        grid[i][j] = max(grid[i - 1][j], grid[i][j - 1])\n")


class TestContamination:
    def test_exact_solution_is_caught(self):
        index = BenchmarkIndex(threshold=4)
        index.add(SOLUTION, "mbpp/1")
        assert index.contaminated("import sys\n\n" + SOLUTION + "\nprint(1)\n")

    def test_reformatting_does_not_hide_it(self):
        """Shingles are over tokens, so indentation and blank lines fall out."""
        index = BenchmarkIndex(threshold=4)
        index.add(SOLUTION, "mbpp/1")
        reflowed = "\n\n".join("    " + line.strip() if line.strip() else line
                               for line in SOLUTION.splitlines())
        assert index.contaminated(reflowed)

    def test_unrelated_code_is_not_flagged(self):
        index = BenchmarkIndex()
        index.add(SOLUTION, "mbpp/1")
        assert not index.contaminated(ORDINARY)

    def test_generic_algorithm_idioms_do_not_trigger(self):
        """The regression that motivated per-problem scoring.

        Counting raw shingles flagged ordinary dynamic-programming files because they
        share ``grid[i][j] = max(grid[i - 1][j]`` and small integer lists with benchmark
        solutions. One or two shingles against a scatter of problems is not evidence.
        """
        index = BenchmarkIndex(threshold=4)
        index.add(SOLUTION, "mbpp/1")
        assert index.best(GENERIC)[1] < index.threshold

    def test_literal_lists_are_never_shingled(self):
        tokens = normalize_code("x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]")
        assert list(shingles(tokens, 13, min_identifiers=4)) == []
        assert list(shingles(tokens, 13, min_identifiers=0)) != []

    def test_scoring_is_per_problem_not_pooled(self):
        """Two shingles each against five problems must not add up to a match."""
        index = BenchmarkIndex(width=3, threshold=4, min_identifiers=1)
        for number in range(5):
            index.add("alpha%d beta%d gamma%d delta%d" % (number, number, number, number),
                      "problem/%d" % number)
        probe = " ".join("alpha%d beta%d gamma%d" % (n, n, n) for n in range(5))
        assert sum(index.hits(probe).values()) >= 5
        assert not index.contaminated(probe)

    def test_best_names_the_matched_problem(self):
        index = BenchmarkIndex(threshold=4)
        index.add(SOLUTION, "mbpp/577")
        assert index.best(SOLUTION)[0] == "mbpp/577"
        assert index.best(ORDINARY) == (None, 0)

    def test_first_writer_keeps_a_shared_shingle(self):
        index = BenchmarkIndex(width=3, min_identifiers=1)
        index.add("shared text here now", "first")
        index.add("shared text here now", "second")
        assert set(index.hashes.values()) == {"first"}

    def test_short_solutions_are_undetectable_and_that_is_reported(self):
        """A two-token solution yields no distinctive shingle at all.

        Pinned because it is a real limitation of the method, not a bug: what such a
        file shares with an ordinary one genuinely is the same content, and no
        threshold separates them. The builder's manifest reports the coverage.
        """
        index = BenchmarkIndex()
        index.add("def f():\n    return 1\n", "tiny")
        assert len(index) == 0
        assert not index.contaminated("def f():\n    return 1\n")

    def test_empty_index_never_flags(self):
        assert BenchmarkIndex().hits("anything at all goes here") == {}
        assert not BenchmarkIndex().contaminated(SOLUTION)

    def test_normalize_drops_layout_but_keeps_tokens(self):
        assert normalize_code("x   =\n\n  1") == normalize_code("x = 1")
        assert normalize_code("x = 1") != normalize_code("y = 1")

    def test_shingle_count(self):
        assert len(list(shingles(["a", "b", "c", "d"], 2, min_identifiers=0))) == 3
        assert len(list(shingles(["a", "b"], 3, min_identifiers=0))) == 0


class TestResume:
    """Restart behaviour, tested against the builder's own accumulator.

    Two things must hold across an interruption and neither is visible when it breaks:
    rows already written are not written a second time, and no row is lost. The
    mechanism is that a checkpoint names every part it knows about, so a part on disk
    that no checkpoint claims was written by a run that died before recording it.
    """

    @staticmethod
    def _corpus(tmp_path):
        build = pytest.importorskip("build")
        return build, build.Corpus(tmp_path, flush_rows=2)

    def _row(self, index, split="train"):
        return {"repo_id": "owner/repo%d" % index, "path": "/m%d.py" % index,
                "source": "x = %d\n" % index, "token_count": 3, "split": split,
                "blob_id": "%040x" % index, "revision_id": "r%d" % index,
                "license_type": "permissive", "detected_licenses": ["MIT"],
                "length_bytes": 6, "star_events_count": 0, "contamination_hits": 0}

    def test_flush_writes_numbered_parts(self, tmp_path):
        _, corpus = self._corpus(tmp_path)
        for index in range(3):
            corpus.add("train", self._row(index))
        corpus.flush()
        assert corpus.parts["train"] == ["part-00000.parquet"]
        assert (tmp_path / "train" / "part-00000.parquet").exists()

    def test_orphaned_part_is_removed_on_resume(self, tmp_path):
        """A part written after the last checkpoint would otherwise duplicate rows."""
        _, corpus = self._corpus(tmp_path)
        corpus.add("train", self._row(0))
        corpus.flush()
        checkpointed = {name: list(parts) for name, parts in corpus.parts.items()}
        corpus.add("train", self._row(1))
        corpus.flush()  # the run dies here, before recording part-00001
        assert len(list((tmp_path / "train").glob("part-*.parquet"))) == 2

        _, resumed = self._corpus(tmp_path)
        assert resumed.reconcile(checkpointed) == 1
        assert resumed.parts["train"] == ["part-00000.parquet"]
        assert len(list((tmp_path / "train").glob("part-*.parquet"))) == 1

    def test_resume_continues_numbering_without_overwriting(self, tmp_path):
        import pyarrow.parquet as pq

        _, corpus = self._corpus(tmp_path)
        corpus.add("train", self._row(0))
        corpus.flush()
        checkpointed = {name: list(parts) for name, parts in corpus.parts.items()}

        _, resumed = self._corpus(tmp_path)
        resumed.reconcile(checkpointed)
        resumed.add("train", self._row(1))
        resumed.flush()
        assert resumed.parts["train"] == ["part-00000.parquet", "part-00001.parquet"]
        rows = pq.read_table(tmp_path / "train").to_pylist()
        assert sorted(row["blob_id"] for row in rows) == ["%040x" % 0, "%040x" % 1]

    def test_reconcile_on_a_fresh_directory_is_a_no_op(self, tmp_path):
        _, corpus = self._corpus(tmp_path)
        assert corpus.reconcile({}) == 0
        assert corpus.parts == {"train": [], "calibration": [], "heldout": []}

    def test_checkpoint_write_is_atomic(self, tmp_path):
        """Writing in place truncates before it writes; a crash there loses the run."""
        import json

        build, _ = self._corpus(tmp_path)
        target = tmp_path / "state.json"
        build.write_atomic(target, {"shard": 0})
        build.write_atomic(target, {"shard": 3, "row_group": 7})
        assert json.loads(target.read_text())["row_group"] == 7
        assert not list(tmp_path.glob("*.tmp"))
