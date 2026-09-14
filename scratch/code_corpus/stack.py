"""Shared Stack v2 access: metadata streaming, blob retrieval, benchmark index.

The probe and the builder need the same three things, and it matters that they need
the *same* ones -- a probe that measures a different retrieval path than the builder
uses measures nothing. Both import from here.
"""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi, HfFileSystem

from distillkit.code_corpus import (
    BenchmarkIndex, blob_url, decode_source, decompress_blob, verify_blob)

DATASET = "bigcode/the-stack-v2-dedup"
CONFIG = "Python"
BASE = "D:/DeepThought/Projects/HybridModel/student-2b-hf"
SHARDS = 6
COLUMNS = ["blob_id", "path", "repo_name", "language", "detected_licenses",
           "license_type", "src_encoding", "is_vendor", "is_generated",
           "length_bytes", "extension", "star_events_count", "revision_id"]


def dataset_revision() -> str:
    """The dataset commit these rows came from, recorded so the corpus is pinnable."""
    return HfApi().dataset_info(DATASET).sha


def shard_paths() -> list[str]:
    return ["datasets/%s/data/%s/train-%05d-of-%05d.parquet" % (DATASET, CONFIG, index, SHARDS)
            for index in range(SHARDS)]


def iter_row_groups(shard: int, start_group: int = 0, columns=COLUMNS):
    """Yield ``(row_group_index, list_of_records)`` from one shard, lazily.

    Row groups are the resume unit. They are read whole because a row group is the
    smallest thing parquet will decode anyway, and one is around 100k metadata rows
    -- tens of megabytes, not the 1.4 GB shard.
    """
    handle = HfFileSystem().open(shard_paths()[shard], "rb")
    reader = pq.ParquetFile(handle)
    for index in range(start_group, reader.num_row_groups):
        yield index, reader.read_row_group(index, columns=columns).to_pylist()


def row_group_count(shard: int) -> int:
    return pq.ParquetFile(HfFileSystem().open(shard_paths()[shard], "rb")).num_row_groups


class BlobFetcher:
    """Thread-pooled retrieval from the Software Heritage object store.

    One :class:`requests.Session` per worker thread, because a session is not thread
    safe but its connection pool is the entire reason this is not latency-bound: a
    cold connection costs the TLS handshake, and a warm one costs one round trip of
    roughly 250 ms. Retrieval is therefore concurrency-bound rather than
    bandwidth-bound, and the pool size is the only throughput knob that matters.

    Returns ``(record, content_or_None, status)`` where status is one of ``ok``,
    ``missing``, ``decompress``, ``checksum``, ``undecodable`` or ``error:<kind>``.
    Every failure is classified rather than dropped -- a corpus that silently loses
    12% of its blobs looks exactly like one that loses none.
    """

    def __init__(self, workers: int = 64, timeout: float = 30.0, retries: int = 2):
        self.workers = workers
        self.timeout = timeout
        self.retries = retries
        self._local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=4, pool_maxsize=4, max_retries=0)
            session.mount("https://", adapter)
            self._local.session = session
        return session

    def _one(self, record):
        blob_id = record["blob_id"]
        url = blob_url(blob_id)
        last = None
        for attempt in range(self.retries + 1):
            try:
                response = self._session().get(url, timeout=self.timeout)
            except requests.RequestException as error:
                last = "error:%s" % type(error).__name__
                time.sleep(0.5 * (attempt + 1))
                continue
            if response.status_code == 404:
                return record, None, "missing"
            if response.status_code != 200:
                last = "error:http%d" % response.status_code
                time.sleep(0.5 * (attempt + 1))
                continue
            try:
                content = decompress_blob(response.content)
            except Exception:
                return record, None, "decompress"
            if not verify_blob(content, blob_id):
                return record, None, "checksum"
            source = decode_source(content, record.get("src_encoding"))
            if source is None:
                return record, None, "undecodable"
            return record, source, "ok"
        return record, None, last or "error:unknown"

    def map(self, records):
        """Retrieve a batch, yielding results as they complete."""
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            yield from pool.map(self._one, records)


def load_tokenizer():
    """The backbone's own tokenizer. Corpus size is only meaningful in its tokens."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(BASE, local_files_only=True)


def tokenizer_identity(tokenizer) -> dict:
    """Enough to prove a later run used the same vocabulary, without shipping it."""
    import hashlib

    path = Path(BASE) / "tokenizer.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return {"source": BASE, "class": type(tokenizer).__name__,
            "vocab_size": tokenizer.vocab_size, "tokenizer_json_sha256": digest}


#: Per benchmark, the columns that identify a *problem*. Deliberately not the EvalPlus
#: ``test`` column: its extended suites are tens of thousands of generated input
#: literals per problem -- one HumanEval+ row shingles to nearly 39,000 tokens of
#: numeric data -- so indexing them flags any ordinary file containing a long numeric
#: list. MBPP's ``test_list`` is the three human-written assertions and is safe to
#: index; MBPP's ``test`` column is the generated dump and is not.
BENCHMARK_COLUMNS = {
    "evalplus/mbppplus": ("prompt", "code", "test_list"),
    "evalplus/humanevalplus": ("prompt", "canonical_solution"),
}


def benchmark_index() -> tuple[BenchmarkIndex, dict]:
    """Index the MBPP+ and HumanEval+ problem statements and reference solutions.

    What identifies a benchmark problem is its statement and its answer, so those are
    what is indexed. The extended test suites are excluded on purpose -- see
    :data:`BENCHMARK_COLUMNS`; they are machine-generated input data rather than
    problem identity, and shingling them trades a precise matcher for one that flags
    ordinary code that merely contains literal data.
    """
    from datasets import load_dataset

    index = BenchmarkIndex()
    counts = {}
    for name, fields in BENCHMARK_COLUMNS.items():
        rows = load_dataset(name, split="test")
        before = len(index)
        for row in rows:
            # Keyed by task, so a match is scored against one problem rather than
            # against the union of all 542 of them.
            key = "%s:%s" % (name.split("/")[-1], row["task_id"])
            for column in fields:
                value = row.get(column)
                if isinstance(value, str):
                    index.add(value, key)
                elif isinstance(value, (list, tuple)):
                    index.extend((str(item) for item in value), key)
        counts[name] = {"problems": len(rows), "columns": list(fields),
                        "shingles_added": len(index) - before}
    return index, {"sets": counts, "threshold": index.threshold,
                   "shingle_width": index.width,
                   "min_identifiers": index.min_identifiers}
