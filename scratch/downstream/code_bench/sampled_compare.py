"""Paired pass@1 over several sampled runs per model.

    python sampled_compare.py source=<dir-prefix> think=<dir-prefix> --seeds 0 1 2

Each model's pass@1 is its per-problem pass rate averaged over seeds; the difference is
paired by problem, with a bootstrap 95% interval over problems.
"""
import argparse
import json
import random
from pathlib import Path


def rates(prefix, seeds):
    per = {}
    for seed in seeds:
        results = json.loads((Path("%s-s%d" % (prefix, seed)) / "eval_results.json").read_text())
        for r in results["results"]:
            ok = r["base"] == "pass" and r["plus"] == "pass"
            per.setdefault(str(r["task_id"]), []).append(ok)
    return {t: sum(v) / len(v) for t, v in per.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arms", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    arms = [a.split("=", 1) for a in args.arms]
    table = {name: rates(prefix, args.seeds) for name, prefix in arms}
    reference = arms[0][0]
    tasks = sorted(table[reference])
    rng = random.Random(0)
    for name, _ in arms:
        mean = sum(table[name][t] for t in tasks) / len(tasks)
        if name == reference:
            print("%-8s pass@1 %.1f%%" % (name, 100 * mean))
            continue
        diffs = [table[name][t] - table[reference][t] for t in tasks]
        boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(10000))
        print("%-8s pass@1 %.1f%%   vs %s %+.1f [%+.1f, %+.1f]"
              % (name, 100 * mean, reference, 100 * sum(diffs) / len(diffs),
                 100 * boots[250], 100 * boots[9750]))


if __name__ == "__main__":
    main()
