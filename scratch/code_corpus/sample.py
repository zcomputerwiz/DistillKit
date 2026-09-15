"""Deterministic sample of each split, plus the license and size distributions.

A corpus that passes every integrity check can still be the wrong corpus, and the only
way to notice is to look at what is in it. This prints a fixed sample per split -- fixed
so two people reading the report see the same files -- alongside the license mix, which
is not filtered by default and is the one property of this corpus that is a judgement
call rather than a measurement.

    python scratch/code_corpus/sample.py --corpus scratch/code_corpus/v1
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

SPLITS = ("train", "calibration", "heldout")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--per-split", type=int, default=15)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = {}
    for split in SPLITS:
        rows = []
        for part in sorted((args.corpus / split).glob("part-*.parquet")):
            rows += pq.read_table(part, columns=["repo_id", "path", "token_count",
                                                 "license_type", "star_events_count",
                                                 "contamination_hits"]).to_pylist()
        # Ranked by a hash of the file's identity rather than by position or stars, so
        # the sample is reproducible without being the head of the file or a popularity
        # ranking -- either of which would misrepresent what the corpus mostly contains.
        rows.sort(key=lambda row: hashlib.blake2b(
            (row["repo_id"] + row["path"]).encode("utf-8"), digest_size=8).digest())
        tokens = sorted(row["token_count"] for row in rows)
        report[split] = {
            "files": len(rows),
            "repositories": len({row["repo_id"] for row in rows}),
            "model_tokens": sum(tokens),
            "tokens_per_file": {
                "min": tokens[0], "p50": tokens[len(tokens) // 2],
                "p90": tokens[int(len(tokens) * 0.9)], "max": tokens[-1]},
            "license_types": dict(collections.Counter(r["license_type"] for r in rows)),
            "files_with_contamination_hits": sum(bool(r["contamination_hits"]) for r in rows),
            "sample": [{"repo_id": r["repo_id"], "path": r["path"],
                        "tokens": r["token_count"], "license": r["license_type"],
                        "stars": r["star_events_count"]}
                       for r in rows[:args.per_split]],
        }

        print("\n=== %s: %d files, %d repos, %.2fM tokens (median %d tokens/file) ==="
              % (split, report[split]["files"], report[split]["repositories"],
                 report[split]["model_tokens"] / 1e6,
                 report[split]["tokens_per_file"]["p50"]))
        print("license: %s" % report[split]["license_types"])
        for entry in report[split]["sample"]:
            print("  %6d tok  %-11s %s%s"
                  % (entry["tokens"], entry["license"], entry["repo_id"], entry["path"]))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("\nwrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
