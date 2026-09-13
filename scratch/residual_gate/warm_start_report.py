"""Assemble the warm-start comparison: stability across seeds, drift, and the noise floor.

Three questions this pulls together from the scoring records, none of which any single
run can answer on its own:

    stability   are C42 and C43 more alike than A42 and A43, over the layer x
                familiarity grid that defines a routing policy?
    drift       how much of the frozen-stage policy is left in C after co-adaptation?
    resolution  how large is the difference between two runs of the *same* configuration,
                and is it smaller than the differences being compared?

The last is the one that decides what the first two are worth. Gate weights are not
comparable between runs -- a permuted hidden layer is the same function -- so every
similarity here is computed on admission per layer per familiarity bucket, which is what
the gate does rather than how it is parameterised.

    python scratch/residual_gate/warm_start_report.py --output scratch/residual_gate/warm-report.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import statistics
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SKIP = ("all", "reach")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def similarity(left, right):
    """Pearson, RMS and mean absolute difference over shared layer x bucket cells."""
    pairs = []
    for layer in sorted(set(left) & set(right)):
        for bucket in sorted(set(left[layer]) & set(right[layer])):
            if bucket in SKIP:
                continue
            first, second = left[layer][bucket], right[layer][bucket]
            if first is not None and second is not None:
                pairs.append((first, second))
    if len(pairs) < 3:
        return None
    first = np.array([pair[0] for pair in pairs])
    second = np.array([pair[1] for pair in pairs])
    difference = first - second
    return {"cells": len(pairs),
            "pearson": (float(np.corrcoef(first, second)[0, 1])
                        if first.std() and second.std() else None),
            "rms": float(np.sqrt((difference ** 2).mean())),
            "mean_absolute": float(np.abs(difference).mean())}


def stage1_grid(path):
    """The frozen-stage policy, in the same shape the co-adaptation scorer emits.

    evaluate.py already recorded admission by the same familiarity buckets, so the warm
    start's own policy needs no extra GPU pass -- it is the run that produced it.
    """
    report = load(path)
    grid = {}
    for layer, entry in report["gate_distribution"].items():
        buckets = {name: (value["mean"] if value else None)
                   for name, value in entry["by_count"].items()}
        buckets["all"] = entry["all"]["mean"]
        grid[layer] = buckets
    return grid


def endpoint(record, label):
    return record["milestones"][-1]["differences"][label]["content"]


def trajectory(record, label):
    return [(entry["step"], entry["differences"][label]["content"]["mean"])
            for entry in record["milestones"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--stage1", type=Path,
                        default=HERE / "familiarity-lr3e3" / "screen-step284.json")
    parser.add_argument("--gate", type=Path,
                        default=HERE / "gates" / "familiarity-lr3e3-step284.pt")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    screen = {seed: load(args.root / ("warm-%s-screen.json" % seed))
              for seed in ("s42", "s43")}
    confirmation = {seed: load(args.root / ("warm-%s-confirmation.json" % seed))
                    for seed in ("s42", "s43")}
    repeat = load(args.root / "repeat-s43-screen.json")

    with io.open(args.gate, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    report = {"warm_start_gate": str(args.gate), "warm_start_sha256": digest,
              "stage1_reference": str(args.stage1)}

    # --- endpoints -------------------------------------------------------------
    report["endpoints"] = {}
    for corpus, records in (("screen", screen), ("confirmation", confirmation)):
        for seed, record in records.items():
            for label in ("C - A", "C - B", "A - B", "C - C(g=1)", "A - A(g=1)",
                          "C(g=1) - B", "A(g=1) - B"):
                value = record["milestones"][-1]["differences"].get(label)
                if value is None:
                    continue
                report["endpoints"].setdefault(label, {})["%s/%s" % (seed, corpus)] = {
                    "mean": value["content"]["mean"], "t": value["content"]["t"],
                    "ci95": value["content"]["ci95"],
                    "better": value["content"]["better"],
                    "n": value["content"]["n"]}

    # --- class split at the endpoint -------------------------------------------
    report["classes"] = {}
    for seed, record in screen.items():
        final = record["milestones"][-1]["differences"]
        for label in ("C - A", "C - B", "A - B"):
            report["classes"].setdefault(label, {})[seed] = {
                name: value["mean"] for name, value in final[label].items()}

    # --- trajectories ----------------------------------------------------------
    report["trajectories"] = {
        "%s %s" % (seed, label): trajectory(record, label)
        for seed, record in screen.items()
        for label in ("A - B", "C - B", "C - A")}
    report["trajectories"]["s43 repeat A - B"] = trajectory(repeat, "A - B")

    # --- cross-seed policy stability -------------------------------------------
    policies = {}
    for seed, record in screen.items():
        for arm, grid in record["milestones"][-1]["policy"].items():
            policies["%s/%s" % (seed, arm)] = grid
    warm = stage1_grid(args.stage1)
    report["cross_seed_policy"] = {
        "A": similarity(policies["s42/A"], policies["s43/A"]),
        "C": similarity(policies["s42/C"], policies["s43/C"]),
    }
    report["drift_from_stage1"] = {
        "s42/C": similarity(policies["s42/C"], warm),
        "s43/C": similarity(policies["s43/C"], warm),
        # A never saw the frozen policy; its distance is the null the drift is read
        # against, not a claim that A was supposed to be near it.
        "s42/A": similarity(policies["s42/A"], warm),
        "s43/A": similarity(policies["s43/A"], warm),
    }
    report["parameter_drift"] = {
        seed: record["milestones"][-1].get("gate_parameter_drift", {})
        for seed, record in screen.items()}
    report["reach"] = {
        "stage1": {layer: max(abs(value - 1.0)
                              for name, value in buckets.items()
                              if name not in SKIP and value is not None)
                   for layer, buckets in warm.items()},
    }
    for seed, record in screen.items():
        for arm, grid in record["milestones"][-1]["policy"].items():
            report["reach"]["%s/%s" % (seed, arm)] = {
                layer: buckets["reach"] for layer, buckets in grid.items()}
    report["reach_over_training"] = {
        "%s/%s" % (seed, arm): [(entry["step"], entry["policy"][arm]["14"]["reach"])
                                for entry in record["milestones"]]
        for seed, record in screen.items()
        for arm in record["milestones"][-1]["policy"]}

    # --- the noise floor -------------------------------------------------------
    measured = [-0.004525,
                endpoint(screen["s43"], "A - B")["mean"],
                endpoint(repeat, "A - B")["mean"]]
    report["resolution"] = {
        "seed43_A_minus_B_repeats": measured,
        "range": max(measured) - min(measured),
        "mean": statistics.fmean(measured),
        "stdev": statistics.stdev(measured),
        "note": ("three runs of one configuration at one seed, identical training stream "
                 "digest; the spread is what any single-run difference has to beat"),
        "compared_with": {
            "C - A s42 screen": endpoint(screen["s42"], "C - A")["mean"],
            "C - A s43 screen": endpoint(screen["s43"], "C - A")["mean"],
            "A - B s42 screen": endpoint(screen["s42"], "A - B")["mean"],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("C - A endpoint")
    for key, value in report["endpoints"]["C - A"].items():
        print("  %-18s %+.6f  t %+6.2f  better %3d/%d"
              % (key, value["mean"], value["t"], value["better"], value["n"]))
    print("\ncross-seed policy similarity (42 vs 43)")
    for arm, value in report["cross_seed_policy"].items():
        print("  arm %s  pearson %+.4f  rms %.4f  mae %.4f"
              % (arm, value["pearson"], value["rms"], value["mean_absolute"]))
    print("\ndrift from the frozen-stage policy")
    for key, value in report["drift_from_stage1"].items():
        print("  %-8s pearson %+.4f  rms %.4f" % (key, value["pearson"], value["rms"]))
    print("\nsame-config repeats of seed 43 A - B: %s  (range %.6f)"
          % (["%+.6f" % v for v in measured], report["resolution"]["range"]))
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
