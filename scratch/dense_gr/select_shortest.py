"""From several rollouts per prompt, keep the shortest good one; set looping ones aside.

The student thinks at 2-2.4x the source's length whatever the reasoning effort says
(effort_ab.ps1), so the length is learned. Training on its own *shortest* successful
trajectory -- correct where a reference answer exists, otherwise finished and loop-free
-- is rejection sampling with a length preference: the same model, taught which of its
own ways of answering to prefer.

- chosen: per prompt, the shortest acceptable rollout (by generated tokens)
- looping: every rollout whose thought (or non-thinking reply) loops (`loop_tokens`),
  for unlikelihood
- prompts with no acceptable rollout contribute nothing but their loops

    python scratch/dense_gr/select_shortest.py ../capture-data/onpolicy-r6.jsonl \\
        --chosen ../capture-data/onpolicy-r6-short.jsonl --looping ../capture-data/onpolicy-r6-loop.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "downstream" / "math_bench"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_math import boxed, correct  # noqa: E402
from teacher_kl import loop_tokens  # noqa: E402

TEACHER = "D:/DeepThought/Projects/HybridModel/teacher-hf"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--chosen", type=Path, required=True)
    parser.add_argument("--looping", type=Path, required=True)
    parser.add_argument("--loop-share", type=float, default=0.05)
    args = parser.parse_args()
    for path in (args.chosen, args.looping):
        if path.exists():
            raise SystemExit("refusing to overwrite %s" % path)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TEACHER)
    close = tok.convert_tokens_to_ids("</think>")
    groups, looping = {}, []
    for line in open(args.rollouts, encoding="utf-8"):
        row = json.loads(line)
        generated = row["text"][row["prompt_chars"]:]
        ids = np.asarray(tok(generated, add_special_tokens=False)["input_ids"])
        share = float(loop_tokens(ids, 0, close).mean()) if len(ids) else 0.0
        verdict = None
        if row.get("reference") is not None:
            verdict = bool(row["finished"]) and correct(boxed(generated.split("</think>")[-1]),
                                                        row["reference"])
        row.update(loop_share=share, correct=verdict)
        if share >= args.loop_share:
            looping.append(row)
            continue
        prompt = re.sub(r":s\d+$", "", row["doc_id"])
        acceptable = row["finished"] and share == 0 and verdict is not False
        groups.setdefault(prompt, []).append((acceptable, row))
    chosen, all_lengths, kept_lengths = [], [], []
    for prompt, rows in groups.items():
        all_lengths += [r["generated_tokens"] for ok, r in rows if ok]
        good = [r for ok, r in rows if ok]
        if good:
            best = min(good, key=lambda r: r["generated_tokens"])
            chosen.append(best)
            kept_lengths.append(best["generated_tokens"])
    with open(args.chosen, "w", encoding="utf-8") as out:
        for row in chosen:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(args.looping, "w", encoding="utf-8") as out:
        for row in looping:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("%d prompts: %d chosen (median %d tokens, against %d over every acceptable rollout), "
          "%d looping rollouts" % (len(groups), len(chosen), statistics.median(kept_lengths),
                                   statistics.median(all_lengths), len(looping)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
