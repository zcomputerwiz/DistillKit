"""Are the MATH proxy's truncated answers loops or unfinished productive thinking?

merge_proxy.py samples MATH train problems with thinking on and a 1,024-token budget, and
about 45% of answers hit it. This regenerates the same problems with the same sampling
and a larger budget, saves every text, and for each answer that runs past 1,024 tokens
reports whether it finishes within the larger budget, whether it is then right, and how
repetitive it is: the share of repeated 4-grams in its last 512 tokens, and the most
times any one 80-character line recurs.

    python scratch/dense_gr/math_truncation.py --arm long3=<checkpoint> --new 4096 \\
        --output scratch/csa2-eval/math-truncation-long3.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_proxy import PROMPT, boxed, correct, generate, problems  # noqa: E402


def repetition(ids, tail=512):
    """Share of 4-grams in the last `tail` tokens that occurred earlier in that tail."""
    window = ids[-tail:]
    grams = [tuple(window[i:i + 4]) for i in range(len(window) - 3)]
    return 1 - len(set(grams)) / max(len(grams), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, metavar="NAME=CHECKPOINT")
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--budget", type=int, default=1024, help="the proxy's budget")
    parser.add_argument("--new", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    from distillkit.models import Qwen35WidenedForCausalLM

    name, path = args.arm.split("=", 1)
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    model = Qwen35WidenedForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    model.config.use_cache = True
    _, math = problems(args.count)
    prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(problem=q)}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True) for q, _ in math]
    texts, cut = generate(model, tok, prompts, args.new, (0.6, 0.95, 20))
    rows = []
    for text, unfinished, (problem, reference) in zip(texts, cut, math):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        lines = Counter(line.strip()[:80] for line in text.splitlines() if len(line.strip()) > 20)
        rows.append({"tokens": len(ids), "finished": not unfinished, "thought_closed": "</think>" in text,
                     "correct": bool(correct(boxed(text.split("</think>")[-1]), reference)),
                     "repeat_4gram_tail": repetition(ids), "max_line_repeats": max(lines.values(), default=0),
                     "problem": problem, "text": text})
    over = [r for r in rows if r["tokens"] > args.budget]
    loops = [r for r in over if r["repeat_4gram_tail"] > 0.5 or r["max_line_repeats"] >= 5]
    summary = {
        "arm": name, "problems": len(rows), "budget": args.budget, "new": args.new,
        "within_budget_correct": sum(r["correct"] for r in rows if r["tokens"] <= args.budget) / len(rows),
        "correct_any_length": sum(r["correct"] for r in rows) / len(rows),
        "over_budget": len(over),
        "over_budget_finished": sum(r["finished"] for r in over),
        "over_budget_finished_correct": sum(r["correct"] for r in over if r["finished"]),
        "over_budget_looping": len(loops),
        "over_budget_unfinished_at_new": sum(not r["finished"] for r in over),
        "median_tokens": sorted(r["tokens"] for r in rows)[len(rows) // 2],
    }
    args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
