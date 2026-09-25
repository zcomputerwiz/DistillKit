"""Generate MBPP+ and HumanEval+ completions for any checkpoint. No code is executed here.

The same protocol `mbpp_plus/generate.py` used for its arms -- chat template, greedy,
left padding with EOS, a 768-token cap, first fenced block extracted by one rule -- made
independent of those arms so the source model and any trained checkpoint are compared
on identical prompts. `prompt_sha256` must match across checkpoints for a given task.

Execution happens elsewhere (see `run_docker.ps1`): the completions are untrusted model
output and this machine holds the checkpoints.

    python scratch/downstream/code_bench/generate.py --checkpoint ../student-2b-hf \\
        --bench mbpp --output scratch/downstream/code_bench/source-mbpp
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dense_gr"))

import smoke_train  # noqa: E402,F401  (triton metadata shim)
import torch  # noqa: E402

FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:```|\Z)", re.DOTALL)

# MBPP+ is `mbpp_plus/generate.py`'s template exactly, so its old stock arm stays comparable.
TEMPLATES = {
    "mbpp": ("Write a Python function for the following task. "
             "Respond with a single fenced Python code block and no explanation.\n\n"
             "{prompt}\n\nYour function must satisfy this test:\n{test}\n"),
    "humaneval": ("Complete the following Python function. Respond with a single fenced "
                  "Python code block containing the complete function, including its "
                  "imports and signature, and no explanation.\n\n```python\n{prompt}```\n"),
}
DATASETS = {"mbpp": "evalplus/mbppplus", "humaneval": "evalplus/humanevalplus"}


def extract(text: str) -> str:
    """The first fenced block, else the raw text. No repair: a completion that does not
    parse is a result."""
    match = FENCE.search(text)
    return (match.group(1) if match else text).strip()


def render(bench, problem):
    if bench == "mbpp":
        return TEMPLATES[bench].format(prompt=problem["prompt"], test=problem["test_list"][0])
    return TEMPLATES[bench].format(prompt=problem["prompt"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bench", choices=sorted(TEMPLATES), required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "completions.jsonl").exists():
        raise SystemExit("refusing to overwrite %s" % args.output)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    config = json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if config.get("csa2_enabled") or config.get("residual_stream_enabled"):
        from distillkit.models import Qwen35WidenedForCausalLM as Model
    else:
        from transformers import AutoModelForCausalLM as Model
    model = Model.from_pretrained(args.checkpoint, dtype=torch.bfloat16).to(args.device).eval()
    model.config.use_cache = True

    problems = load_dataset(DATASETS[args.bench], split="test")
    if args.limit:
        problems = problems.select(range(args.limit))
    tokenizer.padding_side = "left"
    end_of_text = tokenizer.eos_token_id
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": render(args.bench, p)}], tokenize=False,
        add_generation_prompt=True) for p in problems]

    started = time.monotonic()
    records, lengths, truncated = [], [], 0
    for start in range(0, len(prompts), args.batch_size):
        chunk = prompts[start:start + args.batch_size]
        rows = problems.select(range(start, start + len(chunk)))
        tokens = tokenizer(chunk, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(args.device)
        with torch.inference_mode():
            output = model.generate(**tokens, max_new_tokens=args.max_new_tokens,
                                    do_sample=False, temperature=None, top_p=None,
                                    top_k=None, pad_token_id=end_of_text)
        width = tokens["input_ids"].shape[1]
        for offset, problem in enumerate(rows):
            new = output[offset, width:]
            finished = (new == end_of_text).nonzero()
            length = int(finished[0]) if finished.numel() else int(new.numel())
            completion = tokenizer.decode(new[:length], skip_special_tokens=True)
            lengths.append(length)
            is_truncated = not finished.numel() and length >= args.max_new_tokens
            truncated += int(is_truncated)
            records.append({
                "task_id": problem["task_id"],
                "prompt_sha256": hashlib.sha256(chunk[offset].encode("utf-8")).hexdigest(),
                "prompt_tokens": int(tokens["attention_mask"][offset].sum()),
                "generated_tokens": length, "truncated": bool(is_truncated),
                "raw": completion, "code": extract(completion)})
            if start + offset < 3:
                records[-1]["rendered_prompt"] = chunk[offset]
        print("%s %d/%d  %.0f s" % (args.bench, len(records), len(prompts),
                                    time.monotonic() - started), flush=True)

    elapsed = time.monotonic() - started
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "bench": args.bench,
        "dataset": DATASETS[args.bench], "problems": len(records),
        "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size,
        "decoding": {"do_sample": False, "greedy": True, "padding_side": "left",
                     "pad_token": "eos"},
        "instruction_template": TEMPLATES[args.bench],
        "mean_generated_tokens": sum(lengths) / max(len(lengths), 1),
        "median_generated_tokens": sorted(lengths)[len(lengths) // 2],
        "truncations": truncated, "elapsed_seconds": elapsed,
        "tokens_per_second": sum(lengths) / elapsed}, indent=2), encoding="utf-8")
    with open(args.output / "completions.jsonl", "w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record) + "\n")
    print("wrote %d completions to %s" % (len(records), args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
