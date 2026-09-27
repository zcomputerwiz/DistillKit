"""Sort on-policy rollouts into what to learn from, what to push away from, and the rest.

Round 1 trained on every rollout by the teacher's KL, and the teacher endorses a loop it
is shown, so the loops got stronger. Here:

- clean: finished, no loop, and correct wherever a reference answer exists. The student's
  own good trajectories, trained like any document (cross entropy and KL) -- rejection
  sampled self-distillation.
- looping: a stretch repeating a third time or more (`loop_tokens`: in the thought, or the
  whole reply when there is none) over at least
  `--loop-share` of the generated tokens. Unlikelihood on the repeats, KL elsewhere.
- the rest (cut without looping, finished but wrong) is dropped: it teaches long thinking
  or a wrong answer, and the teacher is no reliable corrector of either.

    python scratch/dense_gr/classify_rollouts.py ../capture-data/onpolicy-r2.jsonl \\
        --clean ../capture-data/onpolicy-r2-clean.jsonl --looping ../capture-data/onpolicy-r2-loop.jsonl
"""
from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--looping", type=Path, required=True)
    parser.add_argument("--loop-share", type=float, default=0.05)
    args = parser.parse_args()
    for path in (args.clean, args.looping):
        if path.exists():
            raise SystemExit("refusing to overwrite %s" % path)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TEACHER)
    close = tok.convert_tokens_to_ids("</think>")
    counts = {}
    with open(args.clean, "w", encoding="utf-8") as clean, \
            open(args.looping, "w", encoding="utf-8") as looping:
        for line in open(args.rollouts, encoding="utf-8"):
            row = json.loads(line)
            start = row.get("prompt_chars")  # round 1 did not record it
            if start is None:
                start = row["text"].rfind("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
            generated = row["text"][start:]
            ids = np.asarray(tok(generated, add_special_tokens=False)["input_ids"])
            share = float(loop_tokens(ids, 0, close).mean()) if len(ids) else 0.0
            verdict = None
            if row.get("reference") is not None:
                answer = boxed(generated.split("</think>")[-1])
                verdict = bool(row["finished"]) and correct(answer, row["reference"])
            if share >= args.loop_share:
                kind, out = "looping", looping
            elif row["finished"] and share == 0 and verdict is not False:
                kind, out = "clean", clean
            else:
                kind, out = ("wrong" if row["finished"] else "cut"), None
            source = str(row.get("source"))
            key = "%s %s" % (source, kind)
            counts[key] = counts.get(key, 0) + 1
            if out is not None:
                row.update(loop_share=share, correct=verdict)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    width = max(len(k) for k in counts)
    for key in sorted(counts):
        print("%-*s %6d" % (width, key, counts[key]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
