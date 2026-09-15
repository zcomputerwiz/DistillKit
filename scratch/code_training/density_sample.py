"""Section 2: look at what the token-density filter actually removed.

``min_bytes_per_token = 1.8`` was added after one retrieved file turned out to be
1,035,623 bytes tokenizing to 1,035,618 tokens. The threshold was picked from that one
observation, which is enough to justify the filter existing and not enough to justify its
value -- if it is quietly deleting ordinary Python at a material rate, the corpus is
biased in a way no integrity check would show.

So this retrieves rejects and prints them for inspection. Sanity check only: the
threshold is not retuned unless ordinary useful Python is clearly going.

    python scratch/code_training/density_sample.py --want 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code_corpus"))

from distillkit.code_corpus import FilterLimits, filter_reason, token_filter_reason
from stack import BlobFetcher, iter_row_groups, load_tokenizer

#: Coarse buckets, applied to the decoded source. Heuristic and only for the printout --
#: the classification that matters is the human one, done by reading the excerpts.
LONG_TOKEN = re.compile(r"[A-Za-z0-9+/=_-]{60,}")
NUMERIC_ROW = re.compile(r"^[\s\[\](),.\-0-9eE]+$")


def categorize(source: str) -> str:
    lines = [line for line in source.splitlines() if line.strip()]
    if not lines:
        return "empty"
    numeric = sum(bool(NUMERIC_ROW.match(line)) for line in lines) / len(lines)
    if numeric > 0.5:
        return "compressed literals / numeric tables"
    if LONG_TOKEN.search(source):
        return "obfuscated / data-like (long opaque tokens)"
    mean_line = sum(len(line) for line in lines) / len(lines)
    if mean_line > 300:
        return "minified code"
    if re.search(r"(?m)^\s*(def|class|import|from)\s", source):
        return "ordinary Python"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--want", type=int, default=20)
    parser.add_argument("--shard", type=int, default=1, help="a shard the corpus did not reach")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--excerpt", type=int, default=160)
    parser.add_argument("--output", type=Path,
                        default=Path("scratch/code_training/manifests/density_sample.json"))
    args = parser.parse_args()

    limits = FilterLimits()
    tokenizer = load_tokenizer()
    fetcher = BlobFetcher(workers=args.workers)

    rejects, examined = [], 0
    for _, rows in iter_row_groups(args.shard):
        candidates = [r for r in rows if not filter_reason(r, limits)]
        for start in range(0, len(candidates), args.batch):
            batch = candidates[start:start + args.batch]
            sources, keep = [], []
            for record, source, status in fetcher.map(batch):
                if status == "ok":
                    sources.append(source)
                    keep.append(record)
            if not sources:
                continue
            encoded = tokenizer(sources, add_special_tokens=False)["input_ids"]
            for record, source, ids in zip(keep, sources, encoded):
                examined += 1
                length = len(source.encode("utf-8"))
                if token_filter_reason(len(ids), length, limits) != "token_dense":
                    continue
                rejects.append({
                    "repo_id": record["repo_name"], "path": record["path"],
                    "bytes": length, "tokens": len(ids),
                    "bytes_per_token": round(length / len(ids), 3),
                    "category": categorize(source),
                    "excerpt": source[:args.excerpt].replace("\n", "\\n"),
                })
                if len(rejects) >= args.want:
                    break
            if len(rejects) >= args.want:
                break
        if len(rejects) >= args.want:
            break

    import collections

    counts = collections.Counter(entry["category"] for entry in rejects)
    print("examined %d retrieved files, %d rejected by min_bytes_per_token=%.1f (%.2f%%)\n"
          % (examined, len(rejects), limits.min_bytes_per_token,
             100 * len(rejects) / max(examined, 1)))
    for entry in rejects:
        print("%.2f b/tok  %7d tok  %-38s %s"
              % (entry["bytes_per_token"], entry["tokens"], entry["category"],
                 (entry["repo_id"] + entry["path"])[:60]))
        print("            %s" % entry["excerpt"][:110])
    print("\ncategories: %s" % dict(counts))
    ordinary = counts["ordinary Python"] / max(len(rejects), 1)
    print("ordinary Python among rejects: %.0f%%" % (100 * ordinary))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"examined": examined, "threshold": limits.min_bytes_per_token,
         "reject_rate": len(rejects) / max(examined, 1),
         "categories": dict(counts), "rejects": rejects}, indent=2), encoding="utf-8")
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
