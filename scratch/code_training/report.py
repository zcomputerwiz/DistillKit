"""Assemble the required tables from the evaluation artifacts.

Three tables and the comparison between two of them: the pre-training four-arm baseline,
the code-training curve across milestones, and the post-training four-arm evaluation on
B_code. The central question -- whether G' and S still correct anything once the backbone
has learned Python itself -- is the difference between the first and third, so it is
computed here rather than left to be read off two tables by eye.

Nothing is recomputed. Every number comes from an artifact written by the scorer, and a
missing artifact is reported as missing rather than silently skipped.

    python scratch/code_training/report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from distillkit.code_classes import CODE_CLASSES, HISTORICAL_CLASSES

ROOT = Path("scratch/code_training")
ARMS = ("stock", "gate", "sidecar", "both")
LABELS = {"stock": "B", "gate": "B+G'", "sidecar": "B+S", "both": "B+G'+S"}
COLUMNS = ("content", "keyword", "operator", "delimiter", "newline", "whitespace",
           "other_punct", "control")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def nll(report, arm, klass):
    entry = report["arms"][arm]["code_classes"].get(klass)
    return entry["nll"] if entry and entry["nll"] is not None else None


def arm_table(report, prefix: str) -> list[str]:
    """Absolute NLL per arm per class, then the delta against the stock arm."""
    lines = ["| Arm | Aggregate | " + " | ".join(
        c.replace("other_punct", "Other Punct").replace("whitespace", "Whitespace/Indent")
        .capitalize() for c in COLUMNS) + " |",
        "|---" * (len(COLUMNS) + 2) + "|"]
    for arm in ARMS:
        if arm not in report["arms"]:
            continue
        cells = ["%.4f" % nll(report, arm, c) if nll(report, arm, c) is not None else "--"
                 for c in COLUMNS]
        lines.append("| %s | %.4f | %s |"
                     % (LABELS[arm].replace("B", prefix), report["arms"][arm]["mean_nll"],
                        " | ".join(cells)))
    lines.append("")
    lines.append("Delta against %s (negative is better):" % prefix)
    lines.append("")
    lines.append("| Arm | Aggregate | " + " | ".join(
        c.replace("other_punct", "Other Punct").replace("whitespace", "Whitespace/Indent")
        .capitalize() for c in COLUMNS) + " |")
    lines.append("|---" * (len(COLUMNS) + 2) + "|")
    base = report["arms"]["stock"]
    for arm in ARMS[1:]:
        if arm not in report["arms"]:
            continue
        cells = []
        for klass in COLUMNS:
            a, b = nll(report, arm, klass), nll(report, "stock", klass)
            cells.append("%+.4f" % (a - b) if a is not None and b is not None else "--")
        lines.append("| %s | %+.4f | %s |"
                     % (LABELS[arm].replace("B", prefix),
                        report["arms"][arm]["mean_nll"] - base["mean_nll"],
                        " | ".join(cells)))
    return lines


def historical_table(report, prefix: str) -> list[str]:
    lines = ["| Arm | " + " | ".join(c.capitalize() for c in HISTORICAL_CLASSES) + " |",
             "|---" * (len(HISTORICAL_CLASSES) + 1) + "|"]
    base = report["arms"]["stock"]["historical_classes"]
    for arm in ARMS[1:]:
        if arm not in report["arms"]:
            continue
        entry = report["arms"][arm]["historical_classes"]
        cells = []
        for klass in HISTORICAL_CLASSES:
            a = entry.get(klass, {}).get("nll")
            b = base.get(klass, {}).get("nll")
            cells.append("%+.4f" % (a - b) if a is not None and b is not None else "--")
        lines.append("| %s | %s |" % (LABELS[arm].replace("B", prefix), " | ".join(cells)))
    return lines


def significance(report) -> list[str]:
    """Paired per-document deltas with their t-statistics, which is what settles 1e-3."""
    lines = ["| Arm | Class | Mean delta | Stderr | t | Docs better |",
             "|---|---|---:|---:|---:|---:|"]
    for arm in ARMS[1:]:
        delta = report["arms"].get(arm, {}).get("delta")
        if not delta:
            continue
        for key in ("all", "content", "keyword", "delimiter", "operator", "newline",
                    "whitespace", "other_punct"):
            entry = delta.get(key)
            if not entry:
                continue
            lines.append("| %s | %s | %+.6f | %.6f | %+.1f | %d/%d |"
                         % (LABELS[arm], key, entry["mean"], entry["stderr"],
                            entry["t"], entry["better"], entry["n"]))
    return lines


def persistence(before, after) -> list[str]:
    """The central comparison: is each module's gain still there after code training?"""
    lines = ["| Module | Class | Gain on B0 | Gain on B_code | Retained |",
             "|---|---|---:|---:|---:|"]
    for arm in ARMS[1:]:
        if arm not in before["arms"] or arm not in after["arms"]:
            continue
        for klass in ("aggregate",) + COLUMNS:
            if klass == "aggregate":
                first = before["arms"][arm]["mean_nll"] - before["arms"]["stock"]["mean_nll"]
                second = after["arms"][arm]["mean_nll"] - after["arms"]["stock"]["mean_nll"]
            else:
                a0, b0 = nll(before, arm, klass), nll(before, "stock", klass)
                a1, b1 = nll(after, arm, klass), nll(after, "stock", klass)
                if None in (a0, b0, a1, b1):
                    continue
                first, second = a0 - b0, a1 - b1
            share = (second / first) if abs(first) > 1e-9 else float("nan")
            lines.append("| %s | %s | %+.4f | %+.4f | %s |"
                         % (LABELS[arm], klass, first, second,
                            "%.0f%%" % (100 * share) if share == share else "--"))
    return lines


def curve(reports) -> list[str]:
    lines = ["| Python tokens | Aggregate NLL | Content | Newline | Whitespace | "
             "Punctuation/Delimiter |", "|---:|---:|---:|---:|---:|---:|"]
    for tokens, report in reports:
        historical = report["arms"]["stock"]["historical_classes"]
        lines.append("| %s | %.4f | %.4f | %.4f | %.4f | %.4f |"
                     % ("{:,}".format(tokens), report["arms"]["stock"]["mean_nll"],
                        historical["content"]["nll"], historical["newline"]["nll"],
                        historical["whitespace"]["nll"], historical["punctuation"]["nll"]))
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "baseline_eval/b0.json")
    parser.add_argument("--post", type=Path, default=ROOT / "post_eval/bcode.json")
    parser.add_argument("--milestones", type=Path, default=ROOT / "post_eval")
    parser.add_argument("--output", type=Path, default=ROOT / "TABLES.md")
    args = parser.parse_args()

    before = load(args.baseline)
    after = load(args.post)
    if before is None:
        raise SystemExit("no baseline at %s" % args.baseline)

    out = ["# Held-out Python: tables", "",
           "Evaluation subset: %d documents, %s tokens, digest `%s`."
           % (before["evaluation_subset"]["documents"],
              "{:,}".format(before["evaluation_subset"]["tokens"]),
              before["evaluation_subset"]["digest"][:16]), "",
           "Corpus `%s`, dataset revision `%s`, split seed %d, tokenizer `%s`."
           % (before["corpus"]["corpus"], before["corpus"]["dataset_revision"][:12],
              before["corpus"]["split_seed"],
              before["corpus"]["tokenizer"]["tokenizer_json_sha256"][:12]), "",
           "## Pre-training baseline (B0)", ""]
    out += arm_table(before, "B0")
    out += ["", "### Historical five-class view (deltas vs B0)", ""]
    out += historical_table(before, "B0")
    out += ["", "### Paired per-document significance", ""]
    out += significance(before)

    if before["arms"].get("gate", {}).get("gate_diagnostics"):
        diagnostics = before["arms"]["gate"]["gate_diagnostics"]
        out += ["", "### G' behaviour on Python", "",
                "Reach per layer: %s" % json.dumps(diagnostics["reach"]), "",
                "| Familiarity bucket | Mean gate | Share of tokens |", "|---|---:|---:|"]
        for key, value in diagnostics["mean_by_familiarity"].items():
            share = diagnostics["familiarity_token_share"].get(key, 0.0)
            out.append("| %s | %.5f | %.1f%% |" % (key, value, 100 * share))
        out += ["", "| Target class | Mean gate |", "|---|---:|"]
        for key, value in diagnostics["mean_by_target_class"].items():
            out.append("| %s | %.5f |" % (key, value))

    found = []
    for name, tokens in (("b0", 0), ("5m", 5_000_000), ("10m", 10_000_000),
                         ("20m", 20_000_000), ("final", 30_725_434)):
        path = (args.baseline if name == "b0"
                else args.milestones / ("%s.json" % name))
        report = load(path)
        if report is not None:
            actual = report.get("trained_tokens", tokens)
            found.append((actual, report))
    if len(found) > 1:
        out += ["", "## Code-training curve (stock checkpoint only)", ""]
        out += curve(found)

    if after is not None:
        out += ["", "## Post-training evaluation (B_code)", ""]
        out += arm_table(after, "B_code")
        out += ["", "### Historical five-class view (deltas vs B_code)", ""]
        out += historical_table(after, "B_code")
        out += ["", "### Paired per-document significance", ""]
        out += significance(after)
        out += ["", "## Persistence: module gain before and after code training", ""]
        out += persistence(before, after)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print("\nwrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
