"""The teacher's own code solutions for KodCode problems, for verification and distillation.

Code is where the student trails the source most (HumanEval+ sampled 38.4% against 43.1%)
and where its sampled answers still loop more (51 of MBPP+'s code answers against 20). The
teacher's own solutions, kept only where they pass the problem's pytest suite, are text the
student can copy whole -- cross entropy and KL agree on it. Problems come from KodCode-V1
with code_prompts.py's filters (instruct style, benchmark similarity at most 0.8, a function
the tests import), none of round 8's; prompts are the teacher's template at medium effort,
thinking 60% of the time, no system turn. Generation is a llama-server running the teacher
GGUF (teacher_generate.py's client). Output rows are verify_code.py's rollouts: the prompt
text, the completion appended, `prompt_chars`, `finished`.

    python scratch/dense_gr/teacher_code.py prompts --count 1500 --output ../capture-data/code-prompts-teacher.jsonl
    python scratch/dense_gr/teacher_code.py generate --prompts ../capture-data/code-prompts-teacher.jsonl \\
        --output ../capture-data/teacher-code-rollouts.jsonl
    python scratch/downstream/code_bench/verify_code.py --prompts ../capture-data/code-prompts-teacher.jsonl \\
        --rollouts ../capture-data/teacher-code-rollouts.jsonl --output ../capture-data/teacher-code-verified.jsonl
"""
import argparse
import ast
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teacher_generate import TEACHER, complete, repetition  # noqa: E402

C = Path(__file__).resolve().parents[3] / "capture-data"


def prompts(args):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    tok = AutoTokenizer.from_pretrained(TEACHER)
    used = {json.loads(line)["doc_id"] for line in open(C / "code-prompts-r8.jsonl", encoding="utf-8")}
    rng = random.Random(args.seed)
    stream = load_dataset("KodCode/KodCode-V1", split="train", streaming=True).shuffle(seed=args.seed, buffer_size=20000)
    rows, skipped = [], 0
    for r in stream:
        if len(rows) >= args.count:
            break
        similarity = r.get("benchmark_similarity")
        doc_id = "kodcode:%s" % r["question_id"]
        if (doc_id in used or r.get("style") != "instruct"
                or (similarity is not None and float(similarity) > args.max_similarity)):
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
                                         add_generation_prompt=True, enable_thinking=thinking,
                                         reasoning_effort="medium")
        if len(tok(prompt, add_special_tokens=False)["input_ids"]) > args.width:
            skipped += 1
            continue
        rows.append({"doc_id": doc_id, "prompt": prompt, "question": question, "tests": r["test"],
                     "source": "kodcode:%s" % r["subset"], "thinking": thinking})
    with open(args.output, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote %d prompts (%d thinking, %d skipped) -> %s"
          % (len(rows), sum(r["thinking"] for r in rows), skipped, args.output))


def generate(args):
    rows = [json.loads(line) for line in open(args.prompts, encoding="utf-8")]
    done = set()
    if args.output.exists():
        done = {r["doc_id"] for r in map(json.loads, open(args.output, encoding="utf-8"))}
    jobs = [r for r in rows if r["doc_id"] not in done]

    def run(row):
        try:
            reply = complete(args.server, row["prompt"], args.max_tokens, args.seed)
        except Exception as error:  # a full shared KV pool or a dropped connection: retried on resume
            print("skipped %s: %s" % (row["doc_id"], error), flush=True)
            return None
        text = reply["content"]
        return {"doc_id": row["doc_id"], "thinking": row["thinking"], "text": row["prompt"] + text,
                "prompt_chars": len(row["prompt"]), "tokens": reply.get("tokens_predicted"),
                "finished": reply.get("stop_type") in ("eos", "word") or bool(reply.get("stopped_eos")),
                "thought_closed": (not row["thinking"]) or "</think>" in text, "max_line_repeats": repetition(text)}

    start, produced = time.time(), 0
    with open(args.output, "a", encoding="utf-8") as out, ThreadPoolExecutor(args.workers) as pool:
        for n, row in enumerate(pool.map(run, jobs), 1):
            if row is None:
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            produced += row["tokens"] or 0
            if n % 32 == 0 or n == len(jobs):
                print("%d/%d  %.0f tok/s" % (n, len(jobs), produced / (time.time() - start)), flush=True)


def inputs(args):
    """Capture input from the verified rollouts: test-passing, finished, thought closed,
    no repeated line five times; split as capture_inputs.split_of on the problem id."""
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "frontier"))
    from capture_inputs import split_of

    tok = AutoTokenizer.from_pretrained(TEACHER)
    end = tok.convert_tokens_to_ids("<|im_end|>")
    counts, n, tokens = {}, 0, 0
    with open(args.output, "w", encoding="utf-8") as out:
        for row in map(json.loads, open(args.verified, encoding="utf-8")):
            counts[row["verified"]] = counts.get(row["verified"], 0) + 1
            if not (row["verified"] == "passed" and row["finished"] and row["thought_closed"]
                    and row["max_line_repeats"] < 5):
                continue
            prompt, completion = row["text"][:row["prompt_chars"]], row["text"][row["prompt_chars"]:]
            ids = (tok(prompt, add_special_tokens=False)["input_ids"]
                   + tok(completion, add_special_tokens=False)["input_ids"] + [end])
            out.write(json.dumps({"doc_id": "tcode:" + row["doc_id"], "split": split_of(row["doc_id"]),
                                  "input_ids": ids}) + "\n")
            n += 1
            tokens += len(ids)
    print("verified %s; %d kept, %d tokens -> %s" % (json.dumps(counts), n, tokens, args.output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    i = sub.add_parser("inputs")
    i.add_argument("--verified", type=Path, default=C / "teacher-code-verified.jsonl")
    i.add_argument("--output", type=Path, default=C / "teacher-code.jsonl")
    p = sub.add_parser("prompts")
    p.add_argument("--count", type=int, default=1500)
    p.add_argument("--thinking-share", type=float, default=0.6)
    p.add_argument("--max-similarity", type=float, default=0.8)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--output", type=Path, required=True)
    g = sub.add_parser("generate")
    g.add_argument("--prompts", type=Path, required=True)
    g.add_argument("--output", type=Path, required=True)
    g.add_argument("--server", default="http://127.0.0.1:8090")
    g.add_argument("--workers", type=int, default=32)
    g.add_argument("--max-tokens", type=int, default=6144)
    g.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    {"prompts": prompts, "generate": generate, "inputs": inputs}[args.command](args)


if __name__ == "__main__":
    main()
