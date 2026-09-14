"""The PLE-style control, assembled at run level rather than document level.

Arm D trains a backbone under a routing policy it cannot change. Whether that is worth
doing is not settled by D beating its own gate ablation -- the PLE programme produced a
memory the backbone plainly used and which still failed to beat a backbone trained
without it. Two comparisons decide it, and both are between distributions over training
runs, because three runs of one configuration were measured 0.0033 nats apart:

    D - B           does training under fixed routing beat ordinary adaptation?
    D - (B + G_S1)  does it beat bolting the same policy onto a stock backbone?

The second is the one with teeth. Post-hoc routing is already known to be worth -0.0115
and -0.0281 nats, so training in the gate's presence has to earn its place against a
control that costs nothing.

    python scratch/residual_gate/redundancy_report.py --output scratch/residual_gate/forensics/redundancy.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Every cell of the design, as (file, key). The endpoint records hold the arm with its
# own gate and, for gated arms, the same weights with admission forced to 1.
CELLS = {
    "B42": [("repeats/B42-orig.json", "nll"), ("repeats/B42r1.json", "nll"),
            ("repeats/B42r2.json", "nll")],
    "D42": [("repeats/D42.json", "nll"), ("repeats/D42r1.json", "nll"),
            ("repeats/D42r2.json", "nll")],
    "D42(g=1)": [("repeats/D42.json", "identity_nll"),
                 ("repeats/D42r1.json", "identity_nll"),
                 ("repeats/D42r2.json", "identity_nll")],
    "B43": [("repeats/B43-m.json", "nll"), ("repeats/B43-r.json", "nll")],
    "D43": [("repeats/D43.json", "nll"), ("repeats/D43r1.json", "nll"),
            ("repeats/D43r2.json", "nll")],
    "D43(g=1)": [("repeats/D43.json", "identity_nll"),
                 ("repeats/D43r1.json", "identity_nll"),
                 ("repeats/D43r2.json", "identity_nll")],
}
# Post-hoc routing is an inference-time intervention on each stock run, so it comes from
# the recombination record rather than from a training run of its own.
POSTHOC = {"B42+S1": ["B42+S1", "B42r1+S1", "B42r2+S1"],
           "B43+S1": ["B43+S1", "B43m+S1", "B43r+S1"]}


def describe(values):
    return {"n": len(values), "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values),
            "spread": max(values) - min(values), "values": values}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cells = {}
    for name, sources in CELLS.items():
        values = []
        for relative, key in sources:
            record = json.loads((HERE / relative).read_text(encoding="utf-8"))
            values.append(record[key]["content"])
        cells[name] = describe(values)

    posthoc = json.loads(
        (HERE / "forensics" / "posthoc-screen.json").read_text(encoding="utf-8"))
    for name, combos in POSTHOC.items():
        cells[name] = describe([posthoc["combos"][combo]["nll"]["content"]
                                for combo in combos])
    # The seed-43 stock arm has a third run whose endpoint was scored in the swap batch
    # rather than the repeat batch; it is the same checkpoint under the same evaluator.
    swaps = json.loads(
        (HERE / "forensics" / "q3q4-swaps-screen.json").read_text(encoding="utf-8"))
    cells["B43"] = describe(sorted(
        cells["B43"]["values"] + [swaps["combos"]["B43+none"]["nll"]["content"]]))

    comparisons = {}
    for seed in ("42", "43"):
        for left, right in (("D", "B"), ("D", "B%s+S1"), ("D", "D%s(g=1)"),
                            ("B%s+S1", "B")):
            left_name = (left % seed if "%s" in left else left + seed)
            right_name = (right % seed if "%s" in right else right + seed)
            comparisons["%s - %s" % (left_name, right_name)] = (
                cells[left_name]["mean"] - cells[right_name]["mean"])

    report = {"cells": cells, "run_level_differences": comparisons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("%-12s %-3s %-11s %-11s %-11s %s" % ("cell", "n", "mean", "min", "max",
                                               "spread"))
    for name in ("B42", "B42+S1", "D42", "D42(g=1)",
                 "B43", "B43+S1", "D43", "D43(g=1)"):
        entry = cells[name]
        print("%-12s %-3d %-11.6f %-11.6f %-11.6f %.6f"
              % (name, entry["n"], entry["mean"], entry["min"], entry["max"],
                 entry["spread"]))
    print()
    for label, value in comparisons.items():
        print("  %-22s %+.6f" % (label, value))
    print("\nwrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
