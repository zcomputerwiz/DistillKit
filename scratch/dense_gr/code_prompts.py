"""Served-format code prompts with pytest suites, for verified code rollouts.

Round 7's DPO pairs came from math and general prompts, where a code answer only had to
finish without looping, and code paid for the loop fix (non-thinking HumanEval+ 44.5% ->
40.9%). KodCode-V1 gives function-level Python problems with the expected signature and
pytest suites, decontaminated against HumanEval and MBPP by its authors; problems it
flags as similar to a benchmark are skipped here as well. Each prompt names the
function(s) the tests import and asks for one fenced block, rendered with the
checkpoint's chat template in thinking mode (60%) or not.

    python scratch/dense_gr/code_prompts.py --checkpoint <ckpt> --count 2500 --output ../capture-data/code-prompts-r8.jsonl
"""
import argparse
import ast
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2500)
    parser.add_argument("--thinking-share", type=float, default=0.6)
    # An embedding cosine to the nearest benchmark item: median 0.66, 90th percentile 0.78
    # on a sample; 0.8 drops the most benchmark-like tenth on top of the authors' own cut.
    parser.add_argument("--max-similarity", type=float, default=0.8)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    rng = random.Random(args.seed)
    stream = load_dataset("KodCode/KodCode-V1", split="train", streaming=True).shuffle(seed=args.seed,
                                                                                     buffer_size=20000)
    rows, subsets, skipped = [], {}, 0
    for r in stream:
        if len(rows) >= args.count:
            break
        similarity = r.get("benchmark_similarity")
        if r.get("style") != "instruct" or (similarity is not None and float(similarity) > args.max_similarity):
            skipped += 1
            continue
        functions = r["test_info"] or []
        if isinstance(functions, str):
            try:
                functions = ast.literal_eval(functions)
            except (ValueError, SyntaxError):
                functions = []
        if not functions or "from solution import" not in r["test"]:
            skipped += 1
            continue
        names = "\n".join("`%s`" % f["function_declaration"].strip() for f in functions)
        question = ("%s\n\nImplement:\n%s\n\nGive the complete solution in a single ```python code block."
                    % (r["question"].strip(), names))
        thinking = rng.random() < args.thinking_share
        prompt = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=thinking)
        if len(tok(prompt, add_special_tokens=False)["input_ids"]) > args.width:
            skipped += 1
            continue
        rows.append({"doc_id": "kodcode:%s" % r["question_id"], "prompt": prompt, "tests": r["test"],
                     "reference": None, "source": "kodcode:%s" % r["subset"], "thinking": thinking})
        subsets[r["subset"]] = subsets.get(r["subset"], 0) + 1
    with open(args.output, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote %d code prompts (%d skipped): %s" % (len(rows), skipped, json.dumps(subsets, sort_keys=True)))


if __name__ == "__main__":
    main()
