"""Compare two trained arms to each other, not each to the stock student.

`independent_eval report` measures every checkpoint against the pre-retrofit model,
which is the right frame for "does this adapter hurt". It refuses a trained reference
on purpose. But the curriculum is built out of *matched pairs* -- two runs identical
but for one configuration block -- and the question those pairs exist to answer is
whether arm A beats arm B, on the same documents, with the pairing preserved.

    python scratch/paired_arms.py A.json B.json [--mode enabled]

Reports A minus B: negative favours A. Same 10,000-resample paired percentile
bootstrap over documents, token-weighted the same way, so the numbers sit alongside
the report's own.
"""

import argparse
import json
from pathlib import Path

from distillkit.independent_eval import paired_interval


def load(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not result["complete"]:
        raise ValueError(f"{path} is an incomplete evaluation")
    return result


def compare(a, b, mode):
    if a["split"] != b["split"] or a["tokenizer_sha256"] != b["tokenizer_sha256"]:
        raise ValueError("arms were scored on different splits or tokenizers")
    rows = {}
    for task in sorted(set(a["records"]) & set(b["records"])):
        left, right = a["records"][task], b["records"][task]
        if a["task_sha256"][task] != b["task_sha256"][task] or \
                [r["id"] for r in left] != [r["id"] for r in right]:
            raise ValueError(f"{task}: the two arms did not score the same records")
        if task == "nll":
            tokens = [r["modes"]["enabled"]["tokens"] for r in left]
            if tokens != [r["modes"]["enabled"]["tokens"] for r in right]:
                raise ValueError("causal target counts differ between arms")
            rows[task] = paired_interval([r["modes"][mode]["sum_nll"] for r in left],
                                         [r["modes"][mode]["sum_nll"] for r in right],
                                         tokens)
        else:
            for metric in ("acc", "acc_token_norm", "acc_char_norm"):
                rows[f"{task}/{metric}"] = paired_interval(
                    [r["modes"][mode][metric] for r in left],
                    [r["modes"][mode][metric] for r in right])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--mode", default="enabled")
    parser.add_argument("--output")
    args = parser.parse_args()

    a, b = load(args.left), load(args.right)
    rows = compare(a, b, args.mode)
    left_name, right_name = Path(a["checkpoint"]).name, Path(b["checkpoint"]).name
    print(f"{left_name} minus {right_name}, mode={args.mode} (negative favours "
          f"{left_name})")
    for task, stats in rows.items():
        low, high = stats["ci95"]
        crosses = "" if (low > 0) == (high > 0) else "   (interval spans zero)"
        print("  %-20s %+.6f [%+.6f, %+.6f] over %d documents%s"
              % (task, stats["estimate"], low, high, stats["units"], crosses))
    if args.output:
        Path(args.output).write_text(json.dumps(
            {"left": a["checkpoint"], "right": b["checkpoint"], "mode": args.mode,
             "comparisons": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
