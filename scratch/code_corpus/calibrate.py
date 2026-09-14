"""Calibrate the contamination threshold against real positives and real negatives.

A decontamination threshold picked by intuition is a guess about a number that decides
what leaves the training set, and getting it wrong is invisible in both directions: too
low and the matcher deletes every algorithm-exercise repository, biasing the corpus away
from exactly the code the experiment is about; too high and benchmark answers train the
model that will be evaluated on them.

So it is measured. Positives are the benchmark solutions themselves, in three forms --
verbatim, reformatted, and with identifiers renamed -- because a real leak on GitHub
usually arrives lightly edited. Negatives are ordinary Python actually retrieved from
The Stack, including the algorithm-heavy files that the first matcher flagged. A usable
threshold separates the two distributions with room on both sides.

    python scratch/code_corpus/calibrate.py --corpus scratch/code_corpus/smoke
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pyarrow.parquet as pq

from stack import benchmark_index

WORD = re.compile(r"\b[a-z_][a-z_0-9]{2,}\b")
KEYWORDS = {
    "def", "return", "for", "while", "if", "elif", "else", "import", "from", "class",
    "len", "range", "print", "int", "str", "list", "dict", "set", "tuple", "float",
    "sum", "min", "max", "sorted", "abs", "and", "or", "not", "in", "is", "none",
    "true", "false", "lambda", "with", "as", "try", "except", "append", "self",
}


def reformat(source: str) -> str:
    """Blank lines, doubled indentation, stripped comments -- a plausible copy-edit."""
    lines = []
    for line in source.splitlines():
        stripped = line.split("#")[0].rstrip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        lines.append(" " * indent + stripped.strip())
        lines.append("")
    return "\n".join(lines)


def rename(source: str, seed: int = 0) -> str:
    """Rename local identifiers, leaving keywords and builtins alone."""
    rng = random.Random(seed)
    mapping = {}
    def swap(match):
        word = match.group(0)
        if word in KEYWORDS:
            return word
        mapping.setdefault(word, "v%d" % rng.randrange(10 ** 6))
        return mapping[word]
    return WORD.sub(swap, source)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("scratch/code_corpus/smoke"))
    parser.add_argument("--negatives", type=int, default=600)
    parser.add_argument("--output", type=Path,
                        default=Path("scratch/code_corpus/manifests/calibration.json"))
    args = parser.parse_args()

    from datasets import load_dataset

    index, meta = benchmark_index()
    print("index: %d shingles, width %d, min identifiers %d"
          % (len(index), index.width, index.min_identifiers))

    positives = {"verbatim": [], "reformatted": [], "renamed": []}
    for row in load_dataset("evalplus/mbppplus", split="test"):
        code = row["code"]
        positives["verbatim"].append(index.best(code)[1])
        positives["reformatted"].append(index.best(reformat(code))[1])
        positives["renamed"].append(index.best(rename(code))[1])
    for row in load_dataset("evalplus/humanevalplus", split="test"):
        code = row["prompt"] + row["canonical_solution"]
        positives["verbatim"].append(index.best(code)[1])
        positives["reformatted"].append(index.best(reformat(code))[1])
        positives["renamed"].append(index.best(rename(code))[1])

    negatives = []
    worst = []
    for part in sorted(args.corpus.rglob("part-*.parquet")):
        for row in pq.read_table(part, columns=["repo_id", "path", "source"]).to_pylist():
            key, count = index.best(row["source"])
            negatives.append(count)
            worst.append((count, key, row["repo_id"] + row["path"]))
            if len(negatives) >= args.negatives:
                break
        if len(negatives) >= args.negatives:
            break
    worst.sort(reverse=True)

    def percentiles(values):
        ordered = sorted(values)
        take = lambda q: ordered[min(int(q * len(ordered)), len(ordered) - 1)]
        return {"min": ordered[0], "p50": take(0.5), "p90": take(0.9),
                "p99": take(0.99), "max": ordered[-1]}

    # How much of each problem is even indexable. A two-line solution yields fewer
    # distinctive shingles than any useful threshold, so it cannot be detected in a
    # file that contains it -- and no threshold fixes that, because the content it
    # shares with an ordinary file is genuinely the same content. Reported rather than
    # papered over.
    per_problem = collections.Counter(index.hashes.values())
    sizes = sorted(per_problem.values())
    coverage = {
        "problems_indexed": len(per_problem),
        "problems_total": sum(s["problems"] for s in meta["sets"].values()),
        "shingles_per_problem": percentiles(sizes),
        "detectable_at_threshold": sum(v >= index.threshold for v in per_problem.values()),
    }

    sweep = []
    for candidate in (2, 3, 4, 6, 8, 12, 16, 24):
        sweep.append({
            "threshold": candidate,
            "verbatim_caught": sum(v >= candidate for v in positives["verbatim"]),
            "reformatted_caught": sum(v >= candidate for v in positives["reformatted"]),
            "renamed_caught": sum(v >= candidate for v in positives["renamed"]),
            "ordinary_files_flagged": sum(v >= candidate for v in negatives),
        })

    report = {"index_shingles": len(index), "benchmarks": meta,
              "coverage": coverage, "sweep": sweep,
              "positives": {k: percentiles(v) for k, v in positives.items()},
              "negatives": percentiles(negatives),
              "negatives_examined": len(negatives),
              "highest_scoring_negatives": [
                  {"score": c, "matched": k, "file": f} for c, k, f in worst[:15]]}

    for name, values in positives.items():
        stats = report["positives"][name]
        below = sum(v < index.threshold for v in values)
        report["positives"][name]["below_threshold"] = below
        print("positive %-12s p50 %-5d p10 %-5d min %-4d   %d/%d below threshold %d"
              % (name, stats["p50"], sorted(values)[len(values) // 10], stats["min"],
                 below, len(values), index.threshold))
    above = sum(v >= index.threshold for v in negatives)
    report["negatives"]["above_threshold"] = above
    print("negative (real Python) p50 %d p99 %d max %d   %d/%d above threshold"
          % (report["negatives"]["p50"], report["negatives"]["p99"],
             report["negatives"]["max"], above, len(negatives)))
    print("\n%-10s %-9s %-12s %-8s %s" % ("threshold", "verbatim", "reformatted",
                                          "renamed", "ordinary flagged"))
    for row in sweep:
        print("%-10d %-9d %-12d %-8d %d/%d"
              % (row["threshold"], row["verbatim_caught"], row["reformatted_caught"],
                 row["renamed_caught"], row["ordinary_files_flagged"], len(negatives)))
    print("\ncoverage: %d/%d problems yield >= %d shingles (median %d per problem)"
          % (coverage["detectable_at_threshold"], coverage["problems_total"],
             index.threshold, coverage["shingles_per_problem"]["p50"]))

    print("\nhighest-scoring ordinary files:")
    for row in report["highest_scoring_negatives"][:8]:
        print("  %-4d %-24s %s" % (row["score"], str(row["matched"])[:24], row["file"][:70]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nwrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
