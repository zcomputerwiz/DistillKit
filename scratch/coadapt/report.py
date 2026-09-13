"""Read the paired co-adaptation sweep, with content as the decision variable.

Every figure is a paired per-document difference in nats per token over the same 384
held-out documents, so the standard error is over documents and the t is paired. The
primary comparison is the first column; the rest are diagnostics and cannot by themselves
justify continuing.

    python scratch/coadapt/report.py --sweep scratch/coadapt/sweep
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
from pathlib import Path

CLASSES = ("content", "layout", "punctuation", "control")


def load(path):
    with io.open(path, encoding="utf-8") as handle:
        return json.load(handle)["records"]["nll"]


def series(left_records, left_mode, right_records, right_mode, bucket=None):
    """Per-document (left - right) NLL per token, over documents holding both."""
    values = []
    for left, right in zip(left_records, right_records):
        assert left["id"] == right["id"], "the two runs scored different documents"
        a, b = left["modes"][left_mode], right["modes"][right_mode]
        if bucket:
            a, b = a.get("by_class", {}).get(bucket), b.get("by_class", {}).get(bucket)
            if not a or not b or not a["tokens"]:
                continue
        values.append((a["sum_nll"] - b["sum_nll"]) / a["tokens"])
    return values


def summarise(values):
    if not values:
        return None
    mean = statistics.fmean(values)
    error = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
    return {"mean": mean, "stderr": error,
            "t": mean / error if error else float("nan"),
            "better": sum(1 for value in values if value < 0), "n": len(values),
            "ci95": (mean - 1.96 * error, mean + 1.96 * error)}


def cell(stats):
    if stats is None:
        return "%25s" % "--"
    return "%+11.6f (t %+6.2f)" % (stats["mean"], stats["t"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--random-table", type=Path,
                        help="endpoint chimera result, scored like arm A")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    index = json.loads((args.sweep / "index.json").read_text(encoding="utf-8"))
    rows = []
    for entry in index:
        a, b = load(entry["arm_a"]), load(entry["arm_b"])
        row = {"updates": entry["updates"], "tokens": entry["tokens"], "by_class": {}}
        for bucket in (None,) + CLASSES:
            key = bucket or "aggregate"
            row["by_class"][key] = {
                "A_vs_B": summarise(series(a, "enabled", b, "enabled", bucket)),
                "A_vs_off": summarise(series(a, "enabled", a, "bypassed", bucket)),
                "A_vs_wrong": summarise(series(a, "enabled", a, "shuffled", bucket)),
            }
        # What fraction of the nats arm A saves over arm B is content, and what layout.
        saved = {}
        for bucket in CLASSES:
            values = series(a, "enabled", b, "enabled", bucket)
            tokens = sum(r["modes"]["enabled"]["by_class"][bucket]["tokens"]
                         for r in a if bucket in r["modes"]["enabled"]["by_class"])
            saved[bucket] = (statistics.fmean(values) * tokens) if values else 0.0
        total = sum(abs(value) for value in saved.values()) or 1.0
        row["share"] = {bucket: saved[bucket] / total for bucket in CLASSES}
        rows.append(row)

    print("PRIMARY -- content, arm A against the separately trained control\n")
    print("%9s %10s %25s %16s %8s" % ("updates", "tokens", "A_correct - B_control",
                                      "95% CI", "better"))
    for row in rows:
        stats = row["by_class"]["content"]["A_vs_B"]
        print("%9d %10d %25s  [%+.6f, %+.6f] %5d/%d"
              % (row["updates"], row["tokens"], cell(stats),
                 stats["ci95"][0], stats["ci95"][1], stats["better"], stats["n"]))

    print("\nDIAGNOSTICS -- content column repeated for reference\n")
    print("%9s %25s %25s %25s" % ("updates", "content A-B", "content A-off",
                                  "content A-wrong"))
    for row in rows:
        block = row["by_class"]["content"]
        print("%9d %25s %25s %25s" % (row["updates"], cell(block["A_vs_B"]),
                                      cell(block["A_vs_off"]), cell(block["A_vs_wrong"])))

    print("\nBY CLASS, arm A against control (diagnostic; layout cannot justify anything)\n")
    print("%9s %25s %25s %25s %25s" % ("updates", "content", "layout", "punctuation",
                                       "control"))
    for row in rows:
        print("%9d %25s %25s %25s %25s" % (
            row["updates"],
            *[cell(row["by_class"][bucket]["A_vs_B"]) for bucket in CLASSES]))

    print("\nSHARE of the A-over-B nats by class -- the newline trap guard\n")
    for row in rows:
        parts = "  ".join("%s %+5.1f%%" % (bucket, 100 * row["share"][bucket])
                          for bucket in CLASSES)
        print("%9d  %s" % (row["updates"], parts))

    if args.random_table:
        endpoint = load(index[-1]["arm_a"])
        chimera = load(args.random_table)
        print("\nENDPOINT -- learned rows against restored random rows\n")
        for bucket in CLASSES:
            stats = summarise(series(endpoint, "enabled", chimera, "enabled", bucket))
            print("  %-12s %s" % (bucket, cell(stats)))

    if args.output:
        args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
