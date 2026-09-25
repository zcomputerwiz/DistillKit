"""Paired pass@1 between completion sets scored by robust_eval.py.

    python scratch/downstream/code_bench/compare.py source=<dir> finish=<dir> [...]

The first set is the reference. For each other set: pass@1 on base and plus tests, and the
discordant pairs against the reference with an exact two-sided binomial (McNemar) p-value.
"""
import json
import math
import sys
from pathlib import Path


def load(directory):
    results = json.loads((Path(directory) / "eval_results.json").read_text(encoding="utf-8"))
    rows = {str(r["task_id"]): r for r in results["results"]}
    completions = [json.loads(line) for line in
                   open(Path(directory) / "completions.jsonl", encoding="utf-8-sig")]
    truncated = sum(c["truncated"] for c in completions)
    return rows, truncated, len(completions)


def mcnemar(rescued, broken):
    n = rescued + broken
    if not n:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(rescued, broken) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    arms = [arg.split("=", 1) for arg in sys.argv[1:]]
    loaded = {name: load(path) for name, path in arms}
    reference = arms[0][0]
    ref_rows = loaded[reference][0]
    print("%-10s %8s %8s %6s %9s %9s %8s" % ("arm", "base", "plus", "trunc", "rescued", "broken", "p"))
    for name, _ in arms:
        rows, truncated, total = loaded[name]
        base = sum(r["base"] == "pass" for r in rows.values()) / len(rows)
        plus_ok = {t for t, r in rows.items() if r["base"] == "pass" and r["plus"] == "pass"}
        plus = len(plus_ok) / len(rows)
        if name == reference:
            print("%-10s %7.1f%% %7.1f%% %6d" % (name, 100 * base, 100 * plus, truncated))
            continue
        ref_ok = {t for t, r in ref_rows.items() if r["base"] == "pass" and r["plus"] == "pass"}
        rescued, broken = len(plus_ok - ref_ok), len(ref_ok - plus_ok)
        print("%-10s %7.1f%% %7.1f%% %6d %9d %9d %8.3f"
              % (name, 100 * base, 100 * plus, truncated, rescued, broken, mcnemar(rescued, broken)))


if __name__ == "__main__":
    main()
