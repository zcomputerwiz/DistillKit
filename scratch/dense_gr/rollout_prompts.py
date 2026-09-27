"""Prompts for on-policy round 2, rendered as the student will be prompted when served.

Round 1 cut its prompts from training documents, so every one carried the teacher
template's injected "Reasoning effort is set to xhigh" system text, which serving never
sends, and all were thinking mode. Here that text is deleted, a share of prompts are
non-thinking, and half the prompts have a verifiable answer -- GSM8K and MATH *train*
problems with the benchmark's own boxed-answer instruction (MATH-500 is drawn from MATH's
test split) -- so a rollout can be kept for being right, not only for finishing.

    python scratch/dense_gr/rollout_prompts.py --checkpoint <ckpt> --output ../capture-data/prompts-r2.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "downstream" / "math_bench"))

from run_math import PROMPT, boxed  # noqa: E402

TURN = "<|im_start|>assistant\n"
EFFORT = ("Reasoning effort is set to xhigh. Please think carefully through the task, validate "
          "key assumptions, consider plausible alternatives, and prioritize correctness, "
          "consistency, and clarity in the final answer.")
SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus")


def without_effort(text):
    """The document with the injected effort text deleted, its own system prompt kept."""
    text = text.replace("<|im_start|>system\n%s<|im_end|>\n" % EFFORT, "", 1)
    return text.replace("<|im_start|>system\n%s\n\n" % EFFORT, "<|im_start|>system\n", 1)


def with_effort(text):
    """The document as the teacher template renders it in thinking mode: the effort text
    heading its system turn (its own prompt after a blank line), or as the whole turn."""
    header = "<|im_start|>system\n"
    if text.startswith(header):
        return header + EFFORT + "\n\n" + text[len(header):]
    return header + EFFORT + "<|im_end|>\n" + text


def opening(thinking):
    return TURN + ("<think>\n" if thinking else "<think>\n\n</think>\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="for its chat template")
    parser.add_argument("--corpus", nargs="+", default=[], metavar="PATH=COUNT",
                        help="training corpora to cut prompts from, and how many from each")
    parser.add_argument("--exclude", type=Path, default=None)
    parser.add_argument("--gsm8k", type=int, default=1500)
    parser.add_argument("--math", type=int, default=1500)
    parser.add_argument("--non-thinking-share", type=float, default=0.4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-effort", action="store_true",
                        help="render as the teacher template does, effort text in thinking mode "
                             "only; for a student served with that template. Stripping it from "
                             "training raised held-out code NLL (ablate_continuation.ps1).")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    rng = random.Random(args.seed)
    skip = set(json.load(open(args.exclude))) if args.exclude else set()
    rows = []

    def add(doc_id, prompt, reference, source, thinking):
        if len(tok(prompt, add_special_tokens=False)["input_ids"]) <= args.width:
            rows.append(dict(doc_id=doc_id, prompt=prompt, reference=reference, source=source,
                             thinking=thinking))

    verifiable = [("gsm8k:%d" % i, r["question"], r["answer"].split("####")[-1].strip().replace(",", ""))
                  for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="train"))]
    rng.shuffle(verifiable)
    chosen = verifiable[:args.gsm8k]
    math = []
    for subject in SUBJECTS:
        for i, r in enumerate(load_dataset("EleutherAI/hendrycks_math", subject, split="train")):
            answer = boxed(r["solution"])
            if answer is not None:
                math.append(("math:%s:%d" % (subject, i), r["problem"], answer))
    rng.shuffle(math)
    chosen += math[:args.math]
    for doc_id, question, reference in chosen:
        thinking = rng.random() >= args.non_thinking_share
        prompt = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(problem=question)}],
                                         tokenize=False, add_generation_prompt=True,
                                         enable_thinking=thinking)
        add(doc_id, prompt, reference, doc_id.split(":")[0], thinking)

    for spec in args.corpus:
        path, count = spec.rsplit("=", 1)
        pool = []
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            if row.get("split", "train") == "train" and row["doc_id"] not in skip:
                cut = row["text"].rfind(TURN)
                if cut >= 0:
                    pool.append((row["doc_id"], without_effort(row["text"][:cut])))
        rng.shuffle(pool)
        before = len(rows)
        for doc_id, head in pool:
            if len(rows) - before >= int(count):
                break
            thinking = rng.random() >= args.non_thinking_share
            if args.keep_effort and thinking:
                head = with_effort(head)
            add(doc_id, head + opening(thinking), None, Path(path).stem, thinking)

    rng.shuffle(rows)
    with open(args.output, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {}
    for row in rows:
        key = "%s/%s" % (row["source"], "think" if row["thinking"] else "nothink")
        counts[key] = counts.get(key, 0) + 1
    print("wrote %d prompts: %s" % (len(rows), json.dumps(counts, sort_keys=True)))
    # Exactly the thinking prompts carry the effort text when kept, and none otherwise.
    assert all((EFFORT in row["prompt"]) == (args.keep_effort and row["thinking"]) for row in rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
