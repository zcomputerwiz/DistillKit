"""Re-derive the corpus manifest's claims from the stored parquet, not from the builder.

Every number in the manifest came from counters the builder incremented as it went, so
the manifest cannot catch a builder that miscounted -- it would simply report the wrong
number confidently. This reopens the corpus from disk as a training run would, and
recomputes the claims independently: repository disjointness across splits, token counts
re-tokenized from the stored source, duplicate blobs, and that every file's split matches
what its repository name hashes to.

    python scratch/code_corpus/verify.py --corpus scratch/code_corpus/smoke
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pyarrow.parquet as pq

from distillkit.code_corpus import DEFAULT_PROPORTIONS, split_for_repo
from stack import load_tokenizer

SPLITS = ("train", "calibration", "heldout")


def iter_split(root: Path, split: str, columns=None, batch_size: int = 512):
    """Stream one split the way a training loader would: part by part, batched.

    Parts are read in sorted order and never all at once, so a 30M-token corpus opens
    in constant memory.
    """
    for part in sorted((root / split).glob("part-*.parquet")):
        for batch in pq.ParquetFile(part).iter_batches(batch_size=batch_size,
                                                       columns=columns):
            yield from batch.to_pylist()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--retokenize", type=int, default=400,
                        help="files per split to re-tokenize and compare")
    args = parser.parse_args()

    manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    seed = manifest["seed"]
    tokenizer = load_tokenizer()

    repos = {name: set() for name in SPLITS}
    tokens = collections.Counter()
    files = collections.Counter()
    blobs = collections.Counter()
    misassigned = []
    mismatched = []
    empty = 0

    for split in SPLITS:
        sample, checked = [], 0
        for row in iter_split(args.corpus, split):
            files[split] += 1
            tokens[split] += row["token_count"]
            repos[split].add(row["repo_id"])
            blobs[row["blob_id"]] += 1
            if not row["source"].strip():
                empty += 1
            if split_for_repo(row["repo_id"], seed, DEFAULT_PROPORTIONS) != split:
                misassigned.append((row["repo_id"], split))
            if checked < args.retokenize:
                sample.append(row)
                checked += 1
        if sample:
            encoded = tokenizer([row["source"] for row in sample],
                                add_special_tokens=False)["input_ids"]
            for row, ids in zip(sample, encoded):
                if len(ids) != row["token_count"]:
                    mismatched.append((row["blob_id"], row["token_count"], len(ids)))

    overlap = {
        "train/calibration": len(repos["train"] & repos["calibration"]),
        "train/heldout": len(repos["train"] & repos["heldout"]),
        "calibration/heldout": len(repos["calibration"] & repos["heldout"]),
    }
    duplicates = sum(count - 1 for count in blobs.values() if count > 1)

    print("%-12s %9s %9s %14s   manifest agrees" % ("split", "repos", "files", "tokens"))
    agree = True
    for split in SPLITS:
        claimed = manifest["splits"][split]
        matches = (claimed["repositories"] == len(repos[split])
                   and claimed["files"] == files[split]
                   and claimed["model_tokens"] == tokens[split])
        agree &= matches
        print("%-12s %9d %9d %14d   %s"
              % (split, len(repos[split]), files[split], tokens[split],
                 "yes" if matches else "NO -- manifest says %s" % claimed))

    print("\nrepository overlap: %s" % overlap)
    print("duplicate blobs across all splits: %d" % duplicates)
    print("files whose split does not match their repository hash: %d" % len(misassigned))
    print("re-tokenized token_count mismatches: %d" % len(mismatched))
    print("empty stored sources: %d" % empty)
    for row in mismatched[:5]:
        print("  %s stored %d, recomputed %d" % row)
    for row in misassigned[:5]:
        print("  %s stored in %s" % row)

    ok = (agree and not any(overlap.values()) and not misassigned
          and not mismatched and not duplicates and not empty)
    print("\n%s" % ("VERIFIED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
