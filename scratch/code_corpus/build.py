"""Build a deterministic Python corpus from The Stack v2, split by repository.

The pipeline is metadata, split, filter, retrieve, decontaminate, tokenize, write --
and the order is the design rather than an implementation detail. A repository's split
is decided from its name alone before anything is downloaded, so no file's destination
depends on when it was seen or on how full a counter was; and because the destination
is known that early, a file bound for a split that is already satisfied is skipped
before it costs an HTTP request. That is what makes streaming 45M metadata rows to keep
roughly 25k files affordable.

The invariant the splits guarantee is **isolation**, not completeness: no repository
contributes files to more than one split. It is emphatically *not* that every selected
repository is represented completely -- construction stops once the token targets are met,
so almost every repository in the result contributes only some of its files. Isolation is
what the experiment needs; completeness is not available when 45M metadata rows are
streamed to keep about 42,000 files.

Resume is checkpointed periodically within a row group, not only at its boundary. A row
group is ~100k metadata rows and takes minutes to drain: the first full-scale run was
killed at 120 seconds having retrieved 22,000 files with no checkpoint to resume from.
A checkpoint carries ``(shard, row_group, row_offset)`` and names every parquet part it
knows about, so a crash between a flush and its checkpoint is repaired on restart by
deleting the orphaned part rather than by writing its rows twice. Repository assignments
cannot drift across a restart because they are a pure function of the repository name and
the seed.

    python scratch/code_corpus/build.py --name smoke --train 200000 \
        --calibration 20000 --heldout 50000
    python scratch/code_corpus/build.py --name v1 --train 30000000 \
        --calibration 2000000 --heldout 5000000
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pyarrow as pa
import pyarrow.parquet as pq

from distillkit.code_corpus import (
    DEFAULT_PROPORTIONS, DEFAULT_SEED, FilterLimits, SplitPolicy, filter_reason,
    token_filter_reason,
    split_for_repo)
from stack import (CONFIG, DATASET, SHARDS, BlobFetcher, benchmark_index,
                   dataset_revision, iter_row_groups, load_tokenizer, row_group_count,
                   tokenizer_identity)

ROOT = Path("scratch/code_corpus")
SPLITS = ("train", "calibration", "heldout")

SCHEMA = pa.schema([
    ("repo_id", pa.string()), ("path", pa.string()), ("source", pa.string()),
    ("token_count", pa.int32()), ("split", pa.string()), ("blob_id", pa.string()),
    ("revision_id", pa.string()), ("license_type", pa.string()),
    ("detected_licenses", pa.list_(pa.string())), ("length_bytes", pa.int64()),
    ("star_events_count", pa.int64()), ("contamination_hits", pa.int32()),
])


def write_atomic(path: Path, payload) -> None:
    """Write through a temporary file. ``open(path, "w")`` truncates before it writes,
    so an interrupted checkpoint would otherwise leave a zero-byte state file and lose
    the run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


class Corpus:
    """Accumulates rows per split and flushes them to numbered parquet parts."""

    def __init__(self, root: Path, flush_rows: int = 2000):
        self.root = root
        self.flush_rows = flush_rows
        self.pending = {name: [] for name in SPLITS}
        self.parts: dict[str, list[str]] = {name: [] for name in SPLITS}

    def add(self, split: str, row: dict) -> None:
        self.pending[split].append(row)

    def flush(self, split: str | None = None) -> None:
        for name in ([split] if split else SPLITS):
            rows = self.pending[name]
            if not rows:
                continue
            directory = self.root / name
            directory.mkdir(parents=True, exist_ok=True)
            part = "part-%05d.parquet" % len(self.parts[name])
            table = pa.Table.from_pylist(rows, schema=SCHEMA)
            pq.write_table(table, directory / part, compression="zstd")
            self.parts[name].append(part)
            self.pending[name] = []

    def maybe_flush(self) -> None:
        for name in SPLITS:
            if len(self.pending[name]) >= self.flush_rows:
                self.flush(name)

    def reconcile(self, checkpointed: dict[str, list[str]]) -> int:
        """Delete parquet parts that no checkpoint claims.

        A part on disk that the last checkpoint does not name was written by a run
        that died before recording it. Keeping it would duplicate every row it holds
        when the stream is replayed from the checkpoint.
        """
        removed = 0
        for name in SPLITS:
            self.parts[name] = list(checkpointed.get(name, []))
            directory = self.root / name
            if not directory.exists():
                continue
            keep = set(self.parts[name])
            for path in sorted(directory.glob("part-*.parquet")):
                if path.name not in keep:
                    path.unlink()
                    removed += 1
        return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="corpus directory under scratch/code_corpus")
    parser.add_argument("--train", type=int, default=30_000_000, help="target model tokens")
    parser.add_argument("--calibration", type=int, default=2_000_000)
    parser.add_argument("--heldout", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--batch", type=int, default=512, help="blobs retrieved per round")
    parser.add_argument("--min-bytes", type=int, default=64)
    parser.add_argument("--max-bytes", type=int, default=1 << 20)
    parser.add_argument("--max-tokens", type=int, default=32_768)
    parser.add_argument("--min-bytes-per-token", type=float, default=1.8)
    parser.add_argument("--license-types", default="", help="comma separated, empty keeps all")
    parser.add_argument("--overshoot", type=float, default=1.10)
    parser.add_argument("--checkpoint-every", type=int, default=8,
                        help="retrieval rounds between checkpoints")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = ROOT / args.name
    state_path = root / "state.json"
    targets = {"train": args.train, "calibration": args.calibration,
               "heldout": args.heldout}
    limits = FilterLimits(
        min_bytes=args.min_bytes, max_bytes=args.max_bytes,
        max_tokens=args.max_tokens, min_bytes_per_token=args.min_bytes_per_token,
        license_types=frozenset(x for x in args.license_types.split(",") if x) or None)

    policy = SplitPolicy(targets=targets, overshoot=args.overshoot)
    corpus = Corpus(root)
    counters = collections.Counter()
    seen_blobs: set[str] = set()
    contaminated: list[dict] = []
    files = {name: 0 for name in SPLITS}
    shard, group, row, rounds = 0, 0, 0, 0

    if args.resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        shard, group, row = state["shard"], state["row_group"], state.get("row", 0)
        policy.tokens = dict(state["tokens"])
        policy.repos = {k: set(v) for k, v in state["repos"].items()}
        counters.update(state["counters"])
        seen_blobs = set(state["blobs"])
        contaminated = list(state["contaminated"])
        files = dict(state["files"])
        orphans = corpus.reconcile(state["parts"])
        print("resumed at shard %d group %d row %d, %d orphaned parts removed"
              % (shard, group, row, orphans))
    elif root.exists() and any(root.glob("*/part-*.parquet")):
        if not args.resume:
            raise SystemExit(
                "%s already holds parts; pass --resume or choose a new --name" % root)
        # Parts but no checkpoint: a run died before its first checkpoint, so nothing
        # on disk is accounted for and all of it is orphaned. Starting from zero is the
        # only correct reading -- keeping the parts would duplicate every row in them.
        print("no checkpoint found; discarding %d unaccounted parts" % corpus.reconcile({}))

    revision = dataset_revision()
    tokenizer = load_tokenizer()
    contamination, benchmarks = benchmark_index()
    print("benchmark index: %d shingles from %s" % (len(contamination), benchmarks))

    started = time.monotonic()
    retrieved_bytes = 0
    fetcher = BlobFetcher(workers=args.workers)

    def checkpoint(next_shard: int, next_group: int, next_row: int = 0) -> None:
        corpus.flush()
        write_atomic(state_path, {
            "shard": next_shard, "row_group": next_group, "row": next_row,
            "tokens": policy.tokens, "files": files,
            "repos": {k: sorted(v) for k, v in policy.repos.items()},
            "counters": dict(counters), "blobs": sorted(seen_blobs),
            "contaminated": contaminated, "parts": corpus.parts,
        })

    def consume(batch: list[dict]) -> None:
        nonlocal retrieved_bytes
        kept, splits = [], []
        for record, source, status in fetcher.map(batch):
            counters["retrieval_" + status.split(":")[0]] += 1
            if status != "ok":
                continue
            if record["blob_id"] in seen_blobs:
                counters["duplicate"] += 1
                continue
            seen_blobs.add(record["blob_id"])
            split = record["_split"]
            matched, hits = contamination.best(source)
            if hits >= contamination.threshold:
                # Recorded, never silently dropped: an exclusion nobody can inspect is
                # indistinguishable from a matcher that is quietly broken.
                contaminated.append({"repo_id": record["repo_name"], "path": record["path"],
                                     "blob_id": record["blob_id"], "split": split,
                                     "matched": matched, "hits": hits,
                                     "excluded": split != "heldout"})
                if split != "heldout":
                    counters["contaminated_excluded"] += 1
                    continue
                counters["contaminated_heldout_kept"] += 1
            retrieved_bytes += len(source.encode("utf-8"))
            kept.append(source)
            splits.append((record, hits))
        if not kept:
            return
        encoded = tokenizer(kept, add_special_tokens=False)["input_ids"]
        for (record, hits), source, ids in zip(splits, kept, encoded):
            split = record["_split"]
            reason = token_filter_reason(len(ids), len(source.encode("utf-8")), limits)
            if reason:
                counters["filter_" + reason] += 1
                continue
            # Admission is re-checked here, per file, rather than once per batch: a
            # batch of several hundred is already in flight when a split reaches its
            # target, and recording all of it would push the split past ``overshoot``
            # and make that bound decorative.
            if not policy.admits(split, record["repo_name"]):
                counters["skipped_split_full_late"] += 1
                continue
            policy.record(split, record["repo_name"], len(ids))
            files[split] += 1
            corpus.add(split, {
                "repo_id": record["repo_name"], "path": record["path"],
                "source": source, "token_count": len(ids), "split": split,
                "blob_id": record["blob_id"], "revision_id": record["revision_id"],
                "license_type": record["license_type"],
                "detected_licenses": list(record["detected_licenses"] or []),
                "length_bytes": record["length_bytes"],
                "star_events_count": record["star_events_count"],
                "contamination_hits": hits,
            })
        corpus.maybe_flush()

    batch: list[dict] = []
    done = False
    # Tracked explicitly rather than read off the loop variables afterwards: breaking
    # out of the inner loop lets the outer one advance once more, so ``shard_index``
    # at the end names the shard *after* the one actually consumed -- and a checkpoint
    # written from it would skip that shard's first row group on resume.
    resume_at = (shard, group, row)

    def progress(where: tuple[int, int, int]) -> None:
        elapsed = time.monotonic() - started
        print("shard %d group %d row %d  %s  %.0f s  %.0f rows/s  %.0f files/s"
              % (where + (
                  " ".join("%s=%.2fM" % (n[:3], policy.tokens[n] / 1e6) for n in SPLITS),
                  elapsed, counters["rows_seen"] / max(elapsed, 1e-9),
                  counters["retrieval_ok"] / max(elapsed, 1e-9))), flush=True)

    for shard_index in range(shard, SHARDS):
        if done:
            break
        first = shard_index == shard
        for group_index, rows in iter_row_groups(shard_index, group if first else 0):
            # A row group is ~100k metadata rows and can take minutes to drain, so the
            # offset within it is part of the checkpoint. Without it a kill loses every
            # file retrieved since the group started -- which on the first real run was
            # 22,000 files and essentially the whole corpus.
            offset = row if (first and group_index == group) else 0
            for index in range(offset, len(rows)):
                record = rows[index]
                counters["rows_seen"] += 1
                split = split_for_repo(record["repo_name"], args.seed, DEFAULT_PROPORTIONS)
                if not policy.admits(split, record["repo_name"]):
                    counters["skipped_split_full"] += 1
                    continue
                reason = filter_reason(record, limits)
                if reason:
                    counters["filter_" + reason] += 1
                    continue
                record["_split"] = split
                batch.append(record)
                if len(batch) >= args.batch:
                    consume(batch)
                    batch = []
                    rounds += 1
                    if rounds % args.checkpoint_every == 0:
                        resume_at = (shard_index, group_index, index + 1)
                        checkpoint(*resume_at)
                        progress(resume_at)
                if policy.complete:
                    done = True
                    break
            if batch:
                consume(batch)
                batch = []
            resume_at = ((shard_index, group_index, index + 1) if done
                         else (shard_index, group_index + 1, 0))
            checkpoint(*resume_at)
            progress(resume_at)
            if done:
                break
        group = row = 0
    if batch:
        consume(batch)
    checkpoint(*resume_at)

    elapsed = time.monotonic() - started
    overlap = {
        "train/calibration": sorted(policy.repos["train"] & policy.repos["calibration"]),
        "train/heldout": sorted(policy.repos["train"] & policy.repos["heldout"]),
        "calibration/heldout": sorted(policy.repos["calibration"] & policy.repos["heldout"]),
    }
    manifest = {
        "dataset": DATASET, "config": CONFIG, "revision": revision,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint_every_rounds": args.checkpoint_every,
        "seed": args.seed, "proportions": DEFAULT_PROPORTIONS,
        "split_algorithm": "blake2b-64('<seed>:<repo_name>') / 2**64, tiled over "
                           "splits in sorted name order",
        "admission_policy": "a split stops admitting new repositories at its target and "
                            "closes at target * overshoot; assignment never depends on "
                            "targets or arrival order",
        "split_invariant": "repository-level split isolation: no repository contributes "
                           "files to more than one split. NOT repository completeness -- "
                           "construction stops at the token targets, so most repositories "
                           "are represented by only some of their files.",
        "overshoot": args.overshoot,
        "targets": targets,
        "filters": {"min_bytes": limits.min_bytes, "max_bytes": limits.max_bytes,
                    "max_tokens": limits.max_tokens,
                    "min_bytes_per_token": limits.min_bytes_per_token,
                    "exclude_generated": limits.exclude_generated,
                    "exclude_vendor": limits.exclude_vendor,
                    "license_types": sorted(limits.license_types) if limits.license_types else None},
        "tokenizer": tokenizer_identity(tokenizer),
        "benchmarks": benchmarks,
        "contamination": {
            "shingles": len(contamination),
            "threshold": contamination.threshold,
            # How many benchmark problems yield enough distinctive shingles to be
            # detectable at all. A two-line solution does not, and no threshold makes
            # it so; stating the number is the honest alternative to implying the
            # decontamination is exhaustive.
            "problems_detectable": sum(
                count >= contamination.threshold
                for count in collections.Counter(contamination.hashes.values()).values()),
            "problems_total": sum(entry["problems"] for entry in benchmarks["sets"].values()),
        },
        "splits": {name: {"repositories": len(policy.repos[name]), "files": files[name],
                          "model_tokens": policy.tokens[name],
                          "parts": len(corpus.parts[name])} for name in SPLITS},
        "repo_overlap": {k: len(v) for k, v in overlap.items()},
        "repo_overlap_examples": {k: v[:5] for k, v in overlap.items() if v},
        "counters": dict(counters),
        "retrieved_megabytes": retrieved_bytes / 2 ** 20,
        "elapsed_seconds": elapsed,
        "disk_bytes": sum(p.stat().st_size for p in root.rglob("part-*.parquet")),
    }
    write_atomic(root / "manifest.json", manifest)
    write_atomic(root / "contamination.json", contaminated)

    print("\n" + json.dumps(manifest["splits"], indent=2))
    print("repo overlap: %s" % manifest["repo_overlap"])
    print("counters: %s" % dict(counters))
    print("wrote %s" % (root / "manifest.json"))
    return 0 if not any(overlap.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
