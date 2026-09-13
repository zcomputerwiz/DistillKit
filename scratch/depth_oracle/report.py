"""Read the oracle grid as a compute-versus-content frontier.

The decision variable is the most FFN compute that disappears for negligible content
degradation. Aggregate NLL is printed because it is what a careless reading would use,
and because seeing it move while content does not is the point.

    python scratch/depth_oracle/report.py --grid scratch/depth_oracle/grid/grid.json
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path


def rows(report):
    for entry in report["schedules"]:
        delta = entry["delta"]
        yield {
            "band": "%d-%d" % tuple(entry["band"]),
            "layers": len(entry["layers"]),
            "oracle": entry["oracle"],
            "content": delta["content"]["mean"],
            "content_t": delta["content"]["t"],
            "newline": delta.get("fine:newline", {}).get("mean"),
            "whitespace": delta.get("fine:whitespace", {}).get("mean"),
            "punctuation": delta.get("punctuation", {}).get("mean"),
            "control": delta.get("control", {}).get("mean"),
            "aggregate": entry["aggregate_nll"] - entry["aggregate_baseline_nll"],
            "top1": entry["top1_agreement"],
            "ffn": entry["savings"]["ffn_flops_fraction"],
            "total": entry["savings"]["total_flops_fraction"],
        }


def number(value, width=10, places=6):
    return ("%+*.*f" % (width, places, value)) if value is not None else "%*s" % (width, "--")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=0.001,
                        help="content NLL degradation treated as negligible")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with io.open(args.grid, encoding="utf-8") as handle:
        report = json.load(handle)
    table = sorted(rows(report), key=lambda row: (row["oracle"], row["ffn"]))

    flops = report["flops_per_token"]
    print("model %s, %d documents, %d-layer stack" % (
        report["model"], report["documents"], flops["layers"]))
    print("per-token linear FLOPs: MLP %.2e of %.2e total (%.1f%%)\n" % (
        flops["mlp"], flops["total"], 100 * flops["mlp"] / flops["total"]))

    header = ("%-7s %6s %-19s %10s %8s %10s %10s %10s %8s %7s %7s"
              % ("band", "layers", "oracle", "content", "t", "newline",
                 "punct", "aggregate", "top-1", "ffn%", "all%"))
    print(header)
    print("-" * len(header))
    for row in table:
        print("%-7s %6d %-19s %s %8.1f %s %s %s %8.4f %6.1f%% %6.1f%%" % (
            row["band"], row["layers"], row["oracle"], number(row["content"]),
            row["content_t"], number(row["newline"]), number(row["punctuation"]),
            number(row["aggregate"]), row["top1"],
            100 * row["ffn"], 100 * row["total"]))

    print("\nPareto frontier -- most FFN compute removed per level of content damage\n")
    safe = [row for row in table if row["oracle"] != "content"]
    best = {}
    for row in sorted(safe, key=lambda r: -r["ffn"]):
        key = round(max(row["content"], 0.0), 4)
        if key not in best or row["ffn"] > best[key]["ffn"]:
            best[key] = row
    for key in sorted(best):
        row = best[key]
        print("  content %+.6f  ->  %5.1f%% of FFN FLOPs, %4.1f%% of the model   "
              "[%s %s, top-1 %.4f]"
              % (row["content"], 100 * row["ffn"], 100 * row["total"],
                 row["band"], row["oracle"], row["top1"]))

    within = [row for row in safe if row["content"] <= args.budget]
    print("\nWithin a %+.4f content budget: " % args.budget, end="")
    if within:
        top = max(within, key=lambda row: row["ffn"])
        print("%.1f%% of FFN FLOPs (%.1f%% of the model), %s %s, top-1 %.4f"
              % (100 * top["ffn"], 100 * top["total"], top["band"], top["oracle"],
                 top["top1"]))
    else:
        print("nothing. Every schedule costs more content than the budget allows.")

    if args.output:
        args.output.write_text(json.dumps(table, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
