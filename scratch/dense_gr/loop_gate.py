"""The loop gate: math_truncation.py results on the fresh bank, judged against the base.

All arms, the base included, run the same problems (`--bank fresh`: MATH test outside
MATH-500, untouched by every capture and generation input). Leaving out the screen
problems round 4 trained on would not do: those are where the measured arms looped, so
a subset chosen by their outcomes hides exactly what is being measured. Per arm:
within-budget accuracy, and answers still unfinished and looping at the larger budget
(repeated 4-grams over half the tail, or a line five times). An arm passes if it loops no
more than `--loop-slack` above the base and loses no more than `--accuracy-slack` of
within-budget accuracy. Exits 1 when no candidate passes.

    python scratch/dense_gr/loop_gate.py --base scratch/csa2-eval/math-truncation-fresh-base.json \\
        --candidates scratch/csa2-eval/math-truncation-fresh-long4.json ...
"""
import argparse
import json
from pathlib import Path


def looping(row):
    return row["repeat_4gram_tail"] > 0.5 or row["max_line_repeats"] >= 5


def score(path, budget):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data["rows"]
    if data["summary"]["budget"] != budget:
        raise SystemExit("%s was summarized at budget %s, the gate scores %d" % (path, data["summary"]["budget"], budget))
    return {"arm": data["summary"]["arm"], "problems": len(rows), "problem_set": [r["problem"] for r in rows],
            "new": data["summary"]["new"],
            "within_budget_correct": sum(r["correct"] for r in rows if r["tokens"] <= budget) / len(rows),
            "correct_any_length": sum(r["correct"] for r in rows) / len(rows),
            "unfinished": sum(r["tokens"] > budget and not r["finished"] for r in rows),
            "unfinished_looping": sum(r["tokens"] > budget and not r["finished"] and looping(r) for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--loop-slack", type=int, default=3)
    parser.add_argument("--accuracy-slack", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    base = score(args.base, args.budget)
    limit_loops = base["unfinished_looping"] + args.loop_slack
    limit_accuracy = base["within_budget_correct"] - args.accuracy_slack
    print("%d problems; pass: unfinished-looping <= %d, within-budget >= %.1f%%"
          % (base["problems"], limit_loops, 100 * limit_accuracy))
    results = []
    for path in [args.base] + list(args.candidates):
        s = score(path, args.budget)
        if s.pop("problem_set") != base["problem_set"]:
            raise SystemExit("%s ran different problems from the base" % path)
        if s["new"] != base["new"]:  # a shorter generation budget hides unfinished loops
            raise SystemExit("%s generated %s new tokens, the base %s" % (path, s["new"], base["new"]))
        s["pass"] = (path != args.base and s["unfinished_looping"] <= limit_loops
                     and s["within_budget_correct"] >= limit_accuracy)
        results.append(s)
        print("%-16s within budget %5.1f%%  any length %5.1f%%  unfinished %3d  unfinished-looping %3d  %s"
              % (s["arm"], 100 * s["within_budget_correct"], 100 * s["correct_any_length"], s["unfinished"],
                 s["unfinished_looping"], "base" if path == args.base else ("PASS" if s["pass"] else "FAIL")))
    if args.output:
        args.output.write_text(json.dumps({"limits": {"unfinished_looping": limit_loops,
                                                      "within_budget_correct": limit_accuracy},
                                           "results": results}, indent=1), encoding="utf-8")
    return 0 if any(s["pass"] for s in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
