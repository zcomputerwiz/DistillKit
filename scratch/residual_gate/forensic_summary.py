"""Assemble the forensic matrix and answer the decision-tree questions from it.

Every number here comes from the recombination records; nothing is recomputed. The one
piece of analysis it does is the identifiability question -- whether ``g`` and the update
it actually produces, ``g * r``, disagree about how stable routing is across runs -- which
has to be asked of the same layer x familiarity grid for both quantities or it answers
nothing.

    python scratch/residual_gate/forensic_summary.py --output scratch/residual_gate/forensics/summary.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent / "forensics"
SKIP = ("all", "reach")


def load(name):
    path = HERE / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def merge(*reports):
    combos, per_document = {}, {}
    for report in reports:
        if not report:
            continue
        for name, entry in report["combos"].items():
            combos.setdefault(report["split"], {})[name] = entry
            per_document.setdefault(report["split"], {})[name] = entry["per_document"]
    return combos, per_document


def paired(left, right):
    values = [a - b for a, b in zip(left, right)]
    mean = statistics.fmean(values)
    error = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
    return {"mean": mean, "t": mean / error if error else 0.0,
            "ci95": [mean - 1.96 * error, mean + 1.96 * error],
            "better": sum(1 for value in values if value < 0), "n": len(values)}


def flatten(grid, field):
    """One vector over layer x familiarity cells, for a chosen per-bucket quantity."""
    out = []
    for layer in sorted(grid):
        for bucket in sorted(grid[layer]):
            entry = grid[layer][bucket]
            if entry is not None:
                out.append(entry[field])
    return np.array(out)


def similarity(left, right, field):
    first, second = flatten(left, field), flatten(right, field)
    if len(first) != len(second) or len(first) < 3:
        return None
    difference = first - second
    scale = float(np.abs(np.concatenate([first, second])).mean()) + 1e-12
    return {"pearson": (float(np.corrcoef(first, second)[0, 1])
                        if first.std() and second.std() else None),
            "rms": float(np.sqrt((difference ** 2).mean())),
            # Relative, because ||r|| lives on a different scale to g and an absolute
            # RMS cannot be compared between them.
            "relative_rms": float(np.sqrt((difference ** 2).mean()) / scale)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    reports = [load(name) for name in
               ("q1q2-c43-screen.json", "q3q4-swaps-screen.json",
                "s1-portability-screen.json", "s1-strength-screen.json",
                "key-confirmation.json", "s1-strength-confirmation.json")]
    combos, per_document = merge(*reports)

    summary = {"content": {split: {name: entry["nll"]["content"]
                                   for name, entry in sorted(values.items())}
                           for split, values in combos.items()}}

    # --- the decision tree ------------------------------------------------------
    questions = {}
    for split in combos:
        series = per_document[split]

        def difference(left, right):
            if left in series and right in series:
                return paired(series[left]["content"], series[right]["content"])
            return None

        questions.setdefault("Q1 stage-1 rescues the losing run", {})[split] = {
            "C43+S1 - C43+C43": difference("C43+S1", "C43+C43"),
            "C43+none - C43+C43": difference("C43+none", "C43+C43"),
        }
        questions.setdefault("Q3 stage-1 on stock-trained backbones", {})[split] = {
            "B43+S1 - B43+none": difference("B43+S1", "B43+none"),
            "B42+S1 - B42+none": difference("B42+S1", "B42+none"),
        }
        questions.setdefault("Q4 cross-seed swaps", {})[split] = {
            "C42+C43 - C42+C42": difference("C42+C43", "C42+C42"),
            "C43+C42 - C43+C43": difference("C43+C42", "C43+C43"),
            "A42+A43 - A42+A42": difference("A42+A43", "A42+A42"),
            "A43+A42 - A43+A43": difference("A43+A42", "A43+A43"),
        }
    summary["questions"] = {name: {split: {key: value for key, value in entry.items()
                                           if value is not None}
                                   for split, entry in splits.items()}
                            for name, splits in questions.items()}

    # --- strength curves --------------------------------------------------------
    curves = {}
    for split, values in combos.items():
        for name, entry in values.items():
            backbone, gate = entry["backbone"], entry["gate"]
            if gate == "none":
                curves.setdefault("%s/%s+%s" % (split, backbone, backbone), {})[0.0] = \
                    entry["nll"]["content"]
                continue
            curves.setdefault("%s/%s+%s" % (split, backbone, gate), {})[
                entry["strength"]] = entry["nll"]["content"]
    summary["strength_curves"] = {name: dict(sorted(points.items()))
                                  for name, points in sorted(curves.items())}

    # --- identifiability: is g*r steadier than g? -------------------------------
    grids = {}
    for report in reports:
        if not report or report["split"] != "screen":
            continue
        for name, entry in report["combos"].items():
            if "update" in entry:
                grids[name] = entry["update"]
    comparisons = [("A42+A42", "A43+A43"), ("C42+C42", "C43+C43"),
                   ("C42+C42", "A42+A42"), ("C43+C43", "A43+A43")]
    identifiability = {}
    for left, right in comparisons:
        if left not in grids or right not in grids:
            continue
        identifiability["%s vs %s" % (left, right)] = {
            field: similarity(grids[left], grids[right], field)
            for field in ("g", "r", "gr", "delta_r")}
    summary["identifiability"] = identifiability

    # --- class split for the decisive counterfactual ----------------------------
    summary["classes"] = {}
    for split, values in combos.items():
        for name in ("C43+C43", "C43+S1", "C43+none", "B43+S1", "B43+none"):
            if name in values:
                summary["classes"].setdefault(split, {})[name] = values[name]["nll"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Q1  C43 backbone, its own gate vs the frozen-stage gate")
    for split, entry in summary["questions"]["Q1 stage-1 rescues the losing run"].items():
        for label, value in entry.items():
            print("  %-12s %-22s %+.6f (t %+7.2f, %d/%d)"
                  % (split, label, value["mean"], value["t"], value["better"],
                     value["n"]))
    print("\nQ3  frozen-stage gate on backbones that never saw a gate")
    for split, entry in summary["questions"]["Q3 stage-1 on stock-trained backbones"].items():
        for label, value in entry.items():
            print("  %-12s %-22s %+.6f (t %+7.2f)" % (split, label, value["mean"],
                                                      value["t"]))
    print("\nQ4  cross-seed swaps")
    for split, entry in summary["questions"]["Q4 cross-seed swaps"].items():
        for label, value in entry.items():
            print("  %-12s %-22s %+.6f" % (split, label, value["mean"]))
    print("\nQ5  grid similarity: g against the update it produces")
    for pair, fields in identifiability.items():
        parts = " ".join("%s r=%.3f rms%%=%.3f" % (field, value["pearson"],
                                                   value["relative_rms"])
                         for field, value in fields.items() if value)
        print("  %-22s %s" % (pair, parts))
    print("\nwrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
