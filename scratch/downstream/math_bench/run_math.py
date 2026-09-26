"""GSM8K and MATH-500 for any checkpoint, generated and scored here (nothing is executed).

The same decode paths as `code_bench/generate.py` -- `CompiledGreedy` for the CSA2
checkpoints, HF `generate` otherwise -- greedy or Qwen's thinking-mode sampling, thinking
or not. The prompt asks for the final answer in `\\boxed{}`; the last boxed expression is
taken as the answer, and `math_verify` decides equivalence against the reference, so
`(3, \\pi/2)` matches `\\left( 3, \\frac{\\pi}{2} \\right)`. A completion with no boxed
answer is wrong.

    python scratch/downstream/math_bench/run_math.py --checkpoint <ckpt> --bench gsm8k \\
        --output scratch/downstream/math_bench/<name>-gsm8k [--compiled] [--no-thinking]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code_bench"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from generate import CompiledGreedy, pick  # noqa: E402,F401

PROMPT = ("Solve the following math problem. Reason step by step, then give the final "
          "answer in \\boxed{{}}.\n\n{problem}")


def problems(bench):
    from datasets import load_dataset

    if bench == "gsm8k":
        rows = load_dataset("openai/gsm8k", "main", split="test")
        return [(str(i), r["question"], r["answer"].split("####")[-1].strip().replace(",", ""))
                for i, r in enumerate(rows)]
    rows = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [(r["unique_id"], r["problem"], r["answer"]) for r in rows]


def boxed(text):
    """The contents of the last \\boxed{...}, braces balanced, or None."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth, i = 0, start + len("\\boxed{")
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            if depth == 0:
                return text[i:j]
            depth -= 1
    return None


def correct(answer, reference):
    if answer is None:
        return False
    from math_verify import parse, verify

    gold = parse("$%s$" % reference, parsing_timeout=None)
    guess = parse("$%s$" % answer, parsing_timeout=None)
    if gold and guess and verify(gold, guess, timeout_seconds=None):
        return True
    norm = lambda s: re.sub(r"\s+|\\!|\\,|\$", "", s).rstrip(".")
    return norm(answer) == norm(reference)


def self_test():
    assert boxed(r"so \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert correct("18.0", "18") and correct(r"(3, \pi/2)", r"\left( 3, \frac{\pi}{2} \right)")
    assert not correct(r"(3, \pi)", r"\left( 3, \frac{\pi}{2} \right)") and not correct(None, "1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bench", choices=("gsm8k", "math500"), required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    self_test()
    if (args.output / "results.json").exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    sampling = (0.6, 0.95, 20) if args.sample else None

    from transformers import AutoTokenizer

    config = json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    if config.get("csa2_enabled") or config.get("residual_stream_enabled"):
        from distillkit.models import Qwen35WidenedForCausalLM as Model
    else:
        from transformers import AutoModelForCausalLM as Model
    model = Model.from_pretrained(args.checkpoint, dtype=torch.bfloat16).to("cuda").eval()
    model.config.use_cache = True
    items = problems(args.bench)[:args.limit or None]
    tok.padding_side = "left"
    eos = tok.eos_token_id
    prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(problem=q)}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=not args.no_thinking) for _, q, _ in items]
    runner = width_all = None
    if args.compiled:
        longest = max(len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts)
        width_all = -(-longest // 64) * 64
        runner = CompiledGreedy(model, args.batch_size, width_all, args.max_new_tokens, eos,
                                sampling=sampling, seed=args.seed)
    torch.manual_seed(args.seed)
    started, records = time.monotonic(), []
    for start in range(0, len(prompts), args.batch_size):
        chunk = prompts[start:start + args.batch_size]
        if runner is None:
            tokens = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            decoding = (dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20) if args.sample
                        else dict(do_sample=False, temperature=None, top_p=None, top_k=None))
            with torch.inference_mode():
                output = model.generate(**tokens, max_new_tokens=args.max_new_tokens,
                                        pad_token_id=eos, **decoding)
        else:
            filled = chunk + [chunk[0]] * (args.batch_size - len(chunk))
            tokens = tok(filled, return_tensors="pt", padding="max_length", max_length=width_all,
                         add_special_tokens=False).to("cuda")
            output = runner(tokens["input_ids"], tokens["attention_mask"])
        width = tokens["input_ids"].shape[1]
        for offset, (ident, question, reference) in enumerate(items[start:start + len(chunk)]):
            new = output[offset, width:]
            done = (new == eos).nonzero()
            length = int(done[0]) if done.numel() else int(new.numel())
            text = tok.decode(new[:length], skip_special_tokens=True)
            answer = boxed(text)
            records.append(dict(id=ident, reference=reference, answer=answer,
                                correct=correct(answer, reference), tokens=length,
                                truncated=not done.numel() and length >= args.max_new_tokens,
                                raw=text))
        print("%s %d/%d  %.0f s  running accuracy %.3f"
              % (args.bench, len(records), len(items), time.monotonic() - started,
                 sum(r["correct"] for r in records) / len(records)), flush=True)
    summary = dict(checkpoint=str(args.checkpoint), bench=args.bench, n=len(records),
                   accuracy=sum(r["correct"] for r in records) / len(records),
                   no_answer=sum(r["answer"] is None for r in records),
                   truncated=sum(r["truncated"] for r in records),
                   mean_tokens=sum(r["tokens"] for r in records) / len(records),
                   thinking=not args.no_thinking, sampled=args.sample, seed=args.seed,
                   seconds=time.monotonic() - started)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(dict(summary=summary, records=records),
                                                         indent=1), encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
