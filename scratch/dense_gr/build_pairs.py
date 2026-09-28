"""Preference pairs from several rollouts per prompt: shortest good answer over a failure.

Chosen: the prompt's shortest acceptable rollout (correct where a reference answer exists,
else finished and loop-free), as in `select_shortest.py`. Rejected: rollouts of the same
prompt that loop (`loop_tokens` share >= --loop-share) or ran out of tokens without
finishing. Every side is the current model's own sample at served settings, so every
negative is one it plausibly produces -- the failures users would see, not ones induced
by a broken prompt.

    python scratch/dense_gr/build_pairs.py ../capture-data/onpolicy-r6.jsonl ../capture-data/greedy-r6.jsonl \\
        --output ../capture-data/pairs-r7.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
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
    parser.add_argument("rollouts", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loop-share", type=float, default=0.05)
    parser.add_argument("--max-rejected", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TEACHER)
    close = tok.convert_tokens_to_ids("</think>")
    groups = {}
    for path in args.rollouts:
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            prompt, response = row["text"][:row["prompt_chars"]], row["text"][row["prompt_chars"]:]
            ids = np.asarray(tok(response, add_special_tokens=False)["input_ids"])
            share = float(loop_tokens(ids, 0, close).mean()) if len(ids) else 0.0
            verdict = None
            if row.get("reference") is not None:
                verdict = bool(row["finished"]) and correct(boxed(response.split("</think>")[-1]),
                                                            row["reference"])
            if share >= args.loop_share:
                kind = "looping"
            elif not row["finished"]:
                kind = "cut"
            elif share == 0 and verdict is not False:
                kind = "good"
            else:
                kind = "other"
            key = re.sub(r":s\d+$|:greedy$", "", row["doc_id"])
            groups.setdefault(key, {"prompt": prompt, "source": row.get("source"), "rows": []})
            groups[key]["rows"].append((kind, row["generated_tokens"], response))
    pairs, kinds = [], {}
    for key, group in groups.items():
        good = sorted((n, r) for k, n, r in group["rows"] if k == "good")
        bad = [(k, r) for k, n, r in group["rows"] if k == "looping"]
        bad += [(k, r) for k, n, r in group["rows"] if k == "cut"]
        if not good or not bad:
            continue
        for index, (kind, rejected) in enumerate(bad[:args.max_rejected]):
            pairs.append({"pair_id": "%s:%d" % (key, index), "prompt": group["prompt"],
                          "chosen": good[0][1], "rejected": rejected, "rejected_kind": kind,
                          "source": group["source"]})
            kinds[kind] = kinds.get(kind, 0) + 1
    with open(args.output, "w", encoding="utf-8") as out:
        for pair in pairs:
            out.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print("%d prompts, %d pairs (%s) -> %s" % (len(groups), len(pairs), json.dumps(kinds), args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
