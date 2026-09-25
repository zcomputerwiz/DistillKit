"""Have the teacher rewrite the reasoning blocks that second-guess themselves.

The chat corpus carries hedging in its `<think>` blocks -- 169 documents with three or more
"Wait / Actually / let me re-read", 92 of them GLM-5.2 agent traces at 12.6 hedges per
thousand words -- and cross entropy on that text teaches a 2B the style without the
ability to resolve it. Only reasoning blocks are rewritten: tool calls, tool results and
the visible answers stay byte-identical, because they are what the traces are for.

Inside a block, code fences and tool-call spans become numbered placeholders before the
teacher sees it, and a rewrite that loses, repeats or reorders one is rejected, as is one
that grows, shrinks below a third, or still hedges. Rejected documents keep their original
text. The teacher runs with thinking off.

    python scratch/dense_gr/rewrite_hedges.py --input ../capture-data/run5m.jsonl \\
        --output ../capture-data/rewrite-hedges.jsonl [--limit 10]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

HEDGE = re.compile(r"\b(wait|actually|hold on|hmm|let me re-?read|let me reconsider|"
                   r"let me re-?check|let me double[- ]check|but wait)\b", re.I)
THINK = re.compile(r"<think>(.*?)</think>", re.S)
PROTECT = re.compile(r"```.*?```|<tool_call>.*?</tool_call>", re.S)
PLACEHOLDER = re.compile(r"\u27e6P(\d+)\u27e7")
TEACHER = "D:/DeepThought/Projects/HybridModel/teacher-hf"

PROMPT = """Below is reasoning a model wrote while working on a task. Rewrite it as direct, \
confident reasoning.

- Keep every correct fact, step, calculation and conclusion, in the same order.
- Remove hesitation and self-doubt ("Wait", "Actually", "Hmm", "Let me re-read"), repeated \
re-checks of the same thing, and lines of thought that were later corrected: keep only the \
corrected version.
- Do not add information, do not change any conclusion, do not solve anything differently.
- Keep every placeholder such as \u27e6P1\u27e7 exactly once, in the same order.
- Output only the rewritten reasoning.

Reasoning:
<<<
{text}
>>>"""


def protect(text):
    spans = []

    def swap(match):
        spans.append(match.group(0))
        return "\u27e6P%d\u27e7" % len(spans)

    return PROTECT.sub(swap, text), spans


def restore(text, spans):
    numbers = [int(n) for n in PLACEHOLDER.findall(text)]
    if numbers != list(range(1, len(spans) + 1)):
        return None
    return PLACEHOLDER.sub(lambda m: spans[int(m.group(1)) - 1], text)


def acceptable(original, rewritten):
    if rewritten is None or not rewritten.strip():
        return "placeholders"
    ratio = len(rewritten) / max(len(original), 1)
    if ratio > 1.1:
        return "grew"
    if ratio < 0.3:
        return "shrank"
    if len(HEDGE.findall(rewritten)) > 1:
        return "still hedges"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-hedges", type=int, default=2,
                        help="rewrite a reasoning block with at least this many hedges")
    parser.add_argument("--limit", type=int, default=0, help="documents, for a pilot")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)

    rows = [json.loads(line) for line in open(args.input, encoding="utf-8")]
    todo = [r for r in rows if r.get("split", "train") == "train"
            and any(len(HEDGE.findall(b)) >= args.min_hedges for b in THINK.findall(r["text"]))]
    if args.limit:
        todo = todo[:args.limit]
    blocks = []  # (row index in todo, block index, protected text, spans)
    for number, row in enumerate(todo):
        for index, block in enumerate(THINK.findall(row["text"])):
            if len(HEDGE.findall(block)) >= args.min_hedges:
                masked, spans = protect(block)
                blocks.append((number, index, block, masked, spans))
    print("%d documents, %d reasoning blocks to rewrite" % (len(todo), len(blocks)), flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(TEACHER)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER, device_map="auto", dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True)).eval()

    results, started, generated = {}, time.monotonic(), 0
    order = sorted(range(len(blocks)), key=lambda i: len(blocks[i][3]))
    for start in range(0, len(order), args.batch_size):
        chunk = [blocks[i] for i in order[start:start + args.batch_size]]
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(text=b[3].strip())}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False) for b in chunk]
        tokens = tokenizer(prompts, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(model.device)
        budget = int(1.2 * max(len(tokenizer(b[3])["input_ids"]) for b in chunk)) + 32
        with torch.inference_mode():
            output = model.generate(**tokens, max_new_tokens=budget, do_sample=False,
                                    temperature=None, top_p=None, top_k=None,
                                    pad_token_id=tokenizer.eos_token_id)
        width = tokens["input_ids"].shape[1]
        for offset, (number, index, block, masked, spans) in enumerate(chunk):
            text = tokenizer.decode(output[offset, width:], skip_special_tokens=True).strip()
            text = re.sub(r"^<<<\s*|\s*>>>$", "", text).strip()
            generated += int((output[offset, width:] != tokenizer.eos_token_id).sum())
            restored = restore(text, spans)
            verdict = acceptable(block, restored)
            results[(number, index)] = (restored, verdict)
        print("%d/%d blocks  %.0f s  %.1f tok/s"
              % (min(start + args.batch_size, len(order)), len(order),
                 time.monotonic() - started, generated / (time.monotonic() - started)),
              flush=True)

    kept, verdicts = 0, {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for number, row in enumerate(todo):
            mine = {i: results[(n, i)] for (n, i) in results if n == number}
            if any(v is not None for _, v in mine.values()):
                for _, v in mine.values():
                    if v:
                        verdicts[v] = verdicts.get(v, 0) + 1
                continue
            counter = iter(range(10 ** 6))

            def swap(match):
                i = next(counter)
                return "<think>%s</think>" % (mine[i][0] if i in mine else match.group(1))

            text = THINK.sub(swap, row["text"])
            out.write(json.dumps(dict(row, doc_id=row["doc_id"] + ":rw", text=text,
                                      rewritten_from=row["doc_id"])) + "\n")
            kept += 1
    print("kept %d of %d documents; rejected blocks by reason %s -> %s"
          % (kept, len(todo), verdicts, args.output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
