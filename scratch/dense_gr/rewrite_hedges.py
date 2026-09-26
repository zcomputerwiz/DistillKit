"""Have the teacher cut the second-guessing out of reasoning blocks, deleting only.

The chat corpus carries hedging in its `<think>` blocks -- 169 documents with three or more
"Wait / Actually / let me re-read", 92 of them GLM-5.2 agent traces at 12.6 hedges per
thousand words -- and cross entropy on that text teaches a 2B the style without the
ability to resolve it. Only reasoning blocks are edited: tool calls, tool results and the
visible answers stay byte-identical, because they are what the traces are for.

Delete-only, because a free rewrite was tried and drifted: asked to rewrite a Spanish
poem's scansion work, the teacher reasoned about *different verses* than the ones the
answer then gives, which teaches something worse than hedging. So each block is cut into
numbered sentences and the teacher returns only which to delete -- hesitations, repeated
re-checks, attempts it later corrects. Every word that remains is the original's, so the
reasoning cannot drift from the answer. It is also fast: reading is prefill, and the
teacher writes a few numbers rather than the block (the free rewrite ran 3.7 tok/s in
int8, which would have taken 12-16 hours).

Code fences and tool-call spans are never deletable. A block is accepted if it lost at
least one hedge and kept at least 30% of its text; otherwise it stays as it was. The
teacher runs with thinking off.

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
SENTENCE = re.compile(r"(?<=[.!?])(?=\s)")
TEACHER = "D:/DeepThought/Projects/HybridModel/teacher-hf"

PROMPT = """Below is reasoning a model wrote while working on a task, split into numbered \
segments. Choose segments to DELETE so that what remains reads as direct, confident \
reasoning that still reaches the same conclusions.

Delete: hesitation and self-doubt ("Wait", "Actually", "Hmm", "Let me re-read"), re-checks \
that repeat earlier work, and attempts that are abandoned or corrected later (keep the \
corrected version).
Keep: every fact, step, calculation and conclusion the final result depends on, and any \
segment later text refers back to.

Answer with the segment numbers to delete, comma-separated, or "none". Nothing else.

{segments}"""


def segments(block):
    """`(text, deletable)` pieces whose concatenation is exactly `block`."""
    pieces, cursor = [], 0
    for match in PROTECT.finditer(block):
        pieces += _sentences(block[cursor:match.start()])
        pieces.append((match.group(0), False))
        cursor = match.end()
    pieces += _sentences(block[cursor:])
    return pieces


def _sentences(text):
    out = []
    for line in text.splitlines(keepends=True):
        for part in SENTENCE.split(line):
            if not part:
                continue
            # "1." or "2)" alone is a list marker, not a sentence: it joins what follows.
            marker = out and re.fullmatch(r"\s*\(?\d+[.)]\s*", out[-1][0])
            if out and (marker or not part.strip() or not out[-1][0].strip()):
                out[-1] = (out[-1][0] + part, True)  # whitespace rides with a neighbour
            else:
                out.append((part, True))
    return out


def render(pieces):
    return "\n".join("[%d] %s" % (i + 1, text.strip().replace("\n", " "))
                     for i, (text, _) in enumerate(pieces))


def apply(pieces, answer):
    """Kept text, or None when the answer is not a usable list."""
    answer = answer.strip().lower()
    if answer.startswith("none"):
        return None
    numbers = [int(n) for n in re.findall(r"\d+", answer)]
    if not numbers:
        return None
    drop = {n - 1 for n in numbers if 0 < n <= len(pieces) and pieces[n - 1][1]}
    return "".join(text for i, (text, _) in enumerate(pieces) if i not in drop)


def verdict(original, edited):
    if edited is None:
        return "no deletions"
    if len(edited.strip()) < 0.3 * len(original.strip()):
        return "cut too much"
    if len(HEDGE.findall(edited)) >= len(HEDGE.findall(original)):
        return "no hedge removed"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-hedges", type=int, default=2,
                        help="edit a reasoning block with at least this many hedges")
    parser.add_argument("--limit", type=int, default=0, help="documents, for a pilot")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-memory", default=None,
                        help="per-device budget, e.g. 0=22GiB,1=12GiB")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)

    rows = [json.loads(line) for line in open(args.input, encoding="utf-8")]
    todo = [r for r in rows if r.get("split", "train") == "train"
            and any(len(HEDGE.findall(b)) >= args.min_hedges for b in THINK.findall(r["text"]))]
    if args.limit:
        todo = todo[:args.limit]
    blocks = []  # (document, block index, original, pieces)
    for number, row in enumerate(todo):
        for index, block in enumerate(THINK.findall(row["text"])):
            if len(HEDGE.findall(block)) >= args.min_hedges:
                blocks.append((number, index, block, segments(block)))
    print("%d documents, %d reasoning blocks to edit" % (len(todo), len(blocks)), flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(TEACHER)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER, device_map="auto", dtype=torch.bfloat16,
        max_memory=({int(k) if k.isdigit() else k: v for k, v in
                     (part.split("=") for part in args.max_memory.split(","))}
                    if args.max_memory else None),
        quantization_config=BitsAndBytesConfig(load_in_8bit=True)).eval()

    results, started = {}, time.monotonic()
    order = sorted(range(len(blocks)), key=lambda i: len(blocks[i][2]))
    for start in range(0, len(order), args.batch_size):
        chunk = [blocks[i] for i in order[start:start + args.batch_size]]
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(segments=render(b[3]))}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for b in chunk]
        tokens = tokenizer(prompts, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(model.device)
        budget = min(512, 4 * max(len(b[3]) for b in chunk) + 16)
        with torch.inference_mode():
            output = model.generate(**tokens, max_new_tokens=budget, do_sample=False,
                                    temperature=None, top_p=None, top_k=None,
                                    pad_token_id=tokenizer.eos_token_id)
        width = tokens["input_ids"].shape[1]
        for offset, (number, index, block, pieces) in enumerate(chunk):
            answer = tokenizer.decode(output[offset, width:], skip_special_tokens=True)
            edited = apply(pieces, answer)
            results[(number, index)] = (edited, verdict(block, edited), answer.strip())
        print("%d/%d blocks  %.0f s" % (min(start + args.batch_size, len(order)), len(order),
                                        time.monotonic() - started), flush=True)

    kept, reasons, audit = 0, {}, []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for number, row in enumerate(todo):
            mine = {i: v for (n, i), v in results.items() if n == number}
            for i, (edited, why, answer) in mine.items():
                audit.append(dict(doc_id=row["doc_id"], block=i, verdict=why, answer=answer))
                if why:
                    reasons[why] = reasons.get(why, 0) + 1
            accepted = {i: v[0] for i, v in mine.items() if v[1] is None}
            if not accepted:
                continue
            counter = iter(range(10 ** 6))

            def swap(match):
                i = next(counter)
                return "<think>%s</think>" % accepted.get(i, match.group(1))

            text = THINK.sub(swap, row["text"])
            out.write(json.dumps(dict(row, doc_id=row["doc_id"] + ":rw", text=text,
                                      rewritten_from=row["doc_id"])) + "\n")
            kept += 1
    args.output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=1),
                                                     encoding="utf-8")
    print("edited %d of %d documents; blocks left as they were: %s -> %s"
          % (kept, len(todo), reasons, args.output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
