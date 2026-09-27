"""Cheap screen for merged checkpoints, on nothing the benchmarks score.

Choosing blends by HumanEval+ or GSM8K test accuracy would spend those sets: the winner's
score would no longer be a test result. So candidates are ranked here on held-out capture
documents and *train*-split problems no rollout used, and only the finalists see the
benchmarks:

- code NLL: `teacher-cache-expand-code`'s eval split (rose in every round as HumanEval+ fell)
- thinking NLL: `teacher-cache-thinking`'s eval split
- GSM8K train, non-thinking, greedy: accuracy and unboxed answers (the format fix)
- MATH train, thinking, sampled, 1024 new tokens: accuracy and truncations (the loop fix)

    python scratch/dense_gr/merge_proxy.py think=<ckpt> blend=<ckpt> --output merge-proxy.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "downstream" / "code_bench"))
sys.path.insert(0, str(HERE.parents[0] / "downstream" / "math_bench"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import torch  # noqa: E402

from generate import CompiledGreedy  # noqa: E402
from rollout_prompts import SUBJECTS  # noqa: E402
from run_math import PROMPT, boxed, correct  # noqa: E402

D = Path("D:/DeepThought/Projects/HybridModel")


def problems(count, seed=1):
    """GSM8K and MATH train problems that no round-2 rollout prompt used."""
    from datasets import load_dataset

    used = {json.loads(line)["doc_id"] for line in open(D / "capture-data" / "prompts-r2.jsonl", encoding="utf-8")}
    gsm = [(r["question"], r["answer"].split("####")[-1].strip().replace(",", ""))
           for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="train"))
           if "gsm8k:%d" % i not in used]
    math = []
    for subject in SUBJECTS:
        for i, r in enumerate(load_dataset("EleutherAI/hendrycks_math", subject, split="train")):
            answer = boxed(r["solution"])
            if answer is not None and "math:%s:%d" % (subject, i) not in used:
                math.append((r["problem"], answer))
    rng = random.Random(seed)
    return rng.sample(gsm, count), rng.sample(math, count)


def held_nll(model, tokenizer, cache, count=64):
    from cut_cross_entropy import linear_cross_entropy

    from smoke_train import ANSWER_MARKER, EFFORT_PROMPT
    from teacher_kl import CachedTeacher

    encode = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]
    header, effort = encode("<|im_start|>system\n"), encode(EFFORT_PROMPT)
    whole = encode("<|im_start|>system\n%s<|im_end|>\n" % EFFORT_PROMPT)
    heading = header + effort + encode("\n\n")
    held = CachedTeacher(cache, "eval", device="cuda", max_length=1024,
                         answer_marker=encode(ANSWER_MARKER), min_answer_tokens=2,
                         strip_prefix=[(whole, 0, len(whole)), (heading, len(header), len(heading))])
    total = scored = 0.0
    for doc_id in sorted(held.ids)[:count]:
        ids = held.read(doc_id)["input_ids"]
        with torch.no_grad():
            state = model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                use_cache=False).last_hidden_state
            loss = linear_cross_entropy(state, model.lm_head.weight, ids, shift=1, reduction="sum")
        total += float(loss)
        scored += ids.shape[1] - 1
    return total / scored


def generate(model, tokenizer, prompts, new, sampling):
    width = -(-max(len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts) // 64) * 64
    runner = CompiledGreedy(model, 64, width, new, tokenizer.eos_token_id, sampling=sampling, seed=0)
    texts, cut = [], []
    for start in range(0, len(prompts), 64):
        chunk = prompts[start:start + 64]
        batch = tokenizer(chunk + [chunk[0]] * (64 - len(chunk)), return_tensors="pt",
                          padding="max_length", max_length=width, add_special_tokens=False).to("cuda")
        out = runner(batch["input_ids"], batch["attention_mask"])
        for row in range(len(chunk)):
            produced = out[row, width:]
            done = (produced == tokenizer.eos_token_id).nonzero()
            end = int(done[0]) if done.numel() else produced.numel()
            texts.append(tokenizer.decode(produced[:end], skip_special_tokens=True))
            cut.append(not done.numel())
    return texts, cut


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arms", nargs="+", metavar="NAME=CHECKPOINT")
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    from distillkit.models import Qwen35WidenedForCausalLM

    results = json.loads(args.output.read_text()) if args.output.exists() else {}
    gsm, math = problems(args.count)
    for arm in args.arms:
        name, path = arm.split("=", 1)
        if name in results:
            continue
        tok = AutoTokenizer.from_pretrained(path)
        tok.padding_side = "left"
        model = Qwen35WidenedForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
        model.config.use_cache = True
        row = {"code_nll": held_nll(model, tok, D / "teacher-cache-expand-code"),
               "thinking_nll": held_nll(model, tok, D / "teacher-cache-thinking")}
        render = lambda q, thinking: tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(problem=q)}], tokenize=False,
            add_generation_prompt=True, enable_thinking=thinking)
        texts, _ = generate(model, tok, [render(q, False) for q, _ in gsm], 512, None)
        answers = [boxed(t) for t in texts]
        row["gsm8k_nothink"] = sum(correct(a, r) for a, (_, r) in zip(answers, gsm)) / len(gsm)
        row["gsm8k_unboxed"] = sum(a is None for a in answers)
        texts, cut = generate(model, tok, [render(q, True) for q, _ in math], 1024, (0.6, 0.95, 20))
        row["math_think"] = sum(correct(boxed(t.split("</think>")[-1]), r)
                                for t, (_, r) in zip(texts, math)) / len(math)
        row["math_truncated"] = sum(cut)
        results[name] = row
        args.output.write_text(json.dumps(results, indent=1))
        print("%-14s code nll %.4f  thinking nll %.4f  gsm8k nothink %.1f%% (unboxed %d)  "
              "math think %.1f%% (truncated %d)" % (name, row["code_nll"], row["thinking_nll"],
                                                    100 * row["gsm8k_nothink"], row["gsm8k_unboxed"],
                                                    100 * row["math_think"], row["math_truncated"]), flush=True)
        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
