"""Read the checkpoint sweep as a trajectory rather than a pile of JSON.

Every number is a paired per-document difference on the same 384 held-out documents, so
the frozen backbone cancels exactly and the standard error is over documents.

    python scratch/native_table/sweep_report.py --sweep scratch/independent-eval/sweep
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
from pathlib import Path

TOKENS_PER_UPDATE = 880.0     # measured, with evaluation batches excluded


def paired(records, left, right, field=None, bucket=None):
    """Per-document (left - right) NLL per token, and its t statistic."""
    values = []
    for record in records:
        modes = record["modes"]
        if bucket:
            a = modes[left].get("by_class", {}).get(bucket)
            b = modes[right].get("by_class", {}).get(bucket)
            if not a or not b or not a["tokens"]:
                continue
            values.append((a["sum_nll"] - b["sum_nll"]) / a["tokens"])
        else:
            a, b = modes[left], modes[right]
            values.append((a["sum_nll"] - b["sum_nll"]) / a["tokens"])
    mean = statistics.fmean(values)
    error = statistics.stdev(values) / len(values) ** 0.5
    return mean, error, mean / error if error else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--tokens-per-update", type=float, default=TOKENS_PER_UPDATE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    index = json.loads((args.sweep / "index.json").read_text(encoding="utf-8"))
    rows = []
    for entry in index:
        with io.open(entry["result"], encoding="utf-8") as handle:
            records = json.load(handle)["records"]["nll"]
        row = {"updates": entry["updates"],
               "tokens": int(entry["updates"] * args.tokens_per_update)}
        row["on_off"] = paired(records, "enabled", "bypassed")
        if "shuffled" in records[0]["modes"]:
            row["correct_wrong"] = paired(records, "enabled", "shuffled")
            row["wrong_off"] = paired(records, "shuffled", "bypassed")
        for bucket in ("content", "layout"):
            if records[0]["modes"]["enabled"].get("by_class", {}).get(bucket):
                row[bucket] = paired(records, "enabled", "bypassed", bucket=bucket)
        rows.append(row)

    def cell(value):
        if value is None:
            return "%22s" % "--"
        mean, _, t = value
        return "%+11.6f (t %+6.2f)" % (mean, t)

    print("%9s %10s %22s %22s %22s %22s" % (
        "updates", "tokens", "ON - OFF", "correct - wrong", "content ON-OFF",
        "layout ON-OFF"))
    for row in rows:
        print("%9d %10d %s %s %s %s" % (
            row["updates"], row["tokens"], cell(row.get("on_off")),
            cell(row.get("correct_wrong")), cell(row.get("content")),
            cell(row.get("layout"))))

    print("\nwrong context against OFF -- the arm that should stay at zero:")
    for row in rows:
        if "wrong_off" in row:
            mean, error, t = row["wrong_off"]
            print("  %5d updates  %+.6f +- %.6f  t %+6.2f" % (row["updates"], mean, error, t))

    if args.output:
        args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
