"""Thinking traces from the teacher itself, for distillation the student can copy whole.

The student's long MATH thinking loops (math_truncation.py), and the teacher, given such a
loop, continues it: its top-1 at the first repeat of a line is to repeat it 85% of the time,
97% by the sixth (loop_teacher.py). So the clean signal is the teacher's own thinking. This
samples it from a llama.cpp server running the teacher GGUF (`/completion`, raw prompts
rendered with the teacher's template at medium effort), checks each answer, and writes
every trace with its verdict; capture_inputs-style filtering happens downstream.

    python scratch/dense_gr/teacher_generate.py --problems problems.jsonl --output traces.jsonl
    python scratch/dense_gr/teacher_generate.py --sample 32 --output test.jsonl   (MATH train)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_proxy import PROMPT, boxed, correct  # noqa: E402

TEACHER = "D:/DeepThought/Projects/HybridModel/teacher-hf"


def complete(server, prompt, max_tokens, seed):
    body = json.dumps({"prompt": prompt, "n_predict": max_tokens, "temperature": 0.6, "top_p": 0.95,
                       "top_k": 20, "seed": seed, "cache_prompt": False}).encode()
    request = urllib.request.Request(server + "/completion", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.loads(response.read())


def repetition(text):
    lines = [line.strip()[:80] for line in text.splitlines() if len(line.strip()) > 20]
    return max((lines.count(line) for line in set(lines)), default=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", type=Path, help="JSONL with id, problem, answer")
    parser.add_argument("--sample", type=int, default=0, help="instead: this many MATH train problems")
    parser.add_argument("--samples-per-problem", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--server", default="http://127.0.0.1:8090")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--nothink", action="store_true",
                        help="the teacher's non-thinking mode: its template's empty think block, no thought")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TEACHER)
    if args.problems:
        problems = [json.loads(line) for line in open(args.problems, encoding="utf-8")]
    else:
        from merge_proxy import problems as proxy_problems

        _, math = proxy_problems(args.sample, seed=7)
        problems = [{"id": "math-sample:%d" % i, "problem": q, "answer": a} for i, (q, a) in enumerate(math)]
    done = set()
    if args.output.exists():
        done = {(r["id"], r["seed"]) for r in map(json.loads, open(args.output, encoding="utf-8"))}
    jobs = [(p, s) for p in problems for s in range(args.samples_per_problem) if (p["id"], s) not in done]

    def run(job):
        p, seed = job
        prompt = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(problem=p["problem"])}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=not args.nothink,
                                         reasoning_effort="medium")
        try:
            reply = complete(args.server, prompt, args.max_tokens, seed)
        except Exception as error:  # a full shared KV pool or a dropped connection: retried on resume
            print("skipped %s: %s" % (p["id"], error), flush=True)
            return None
        text = reply["content"]
        finished = reply.get("stop_type") in ("eos", "word") or bool(reply.get("stopped_eos"))
        return {"id": p["id"], "seed": seed, "problem": p["problem"], "reference": p["answer"], "prompt": prompt,
                "text": text, "tokens": reply.get("tokens_predicted"), "finished": finished,
                "thought_closed": args.nothink or "</think>" in text,
                "correct": bool(correct(boxed(text.split("</think>")[-1]), p["answer"])),
                "max_line_repeats": repetition(text)}

    start, produced = time.time(), 0
    with open(args.output, "a", encoding="utf-8") as out, ThreadPoolExecutor(args.workers) as pool:
        for n, row in enumerate(pool.map(run, jobs), 1):
            if row is None:
                continue
            out.write(json.dumps(row) + "\n")
            out.flush()
            produced += row["tokens"] or 0
            if n % 16 == 0 or n == len(jobs):
                print("%d/%d  %.0f tok/s" % (n, len(jobs), produced / (time.time() - start)), flush=True)
    rows = [json.loads(line) for line in open(args.output, encoding="utf-8")]
    print(json.dumps({"traces": len(rows), "finished": sum(r["finished"] for r in rows),
                      "correct": sum(r["correct"] for r in rows),
                      "looping (a line 5+ times)": sum(r["max_line_repeats"] >= 5 for r in rows),
                      "median tokens": sorted(r["tokens"] or 0 for r in rows)[len(rows) // 2]}, indent=1))


if __name__ == "__main__":
    main()
