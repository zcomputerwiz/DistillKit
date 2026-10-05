"""Are the teacher's post-thinking answers good non-thinking examples?

The teacher's thinking-mode answers (teacher_generate.py) are full worked solutions after the
thought. Rendered with an empty think block they would teach the student's non-thinking mode
-- if the teacher, scoring them in non-thinking mode, finds them as natural as its own
non-thinking answers. A step it only worked out while thinking shows up as a surprise: a token
its non-thinking self gives low probability. For the same problems this compares the
teacher's own non-thinking answers (teacher_generate.py --nothink) with the post-thinking ones,
on the teacher's non-thinking view of each.

    python scratch/dense_gr/nothink_pilot.py build --count 200     (problems, from kept traces)
    python scratch/dense_gr/teacher_generate.py --nothink --problems ../capture-data/nothink-pilot-problems.jsonl \\
        --output ../capture-data/nothink-pilot-native.jsonl
    python scratch/dense_gr/nothink_pilot.py inputs                 (both, rendered for capture)
    (capture ../capture-data/nothink-pilot.jsonl at 8K into ../teacher-cache-nothink-pilot)
    python scratch/dense_gr/nothink_pilot.py score
"""
import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
C = Path(__file__).resolve().parents[3] / "capture-data"
TEACHER = "D:/DeepThought/Projects/HybridModel/teacher-hf"


def trace_id(row):
    return "%s#%d" % (row["id"], row["seed"])


def build(args):
    keep = set(json.loads((C / "teacher-gen-keep.json").read_text(encoding="utf-8")))
    rows = [r for r in map(json.loads, open(C / "teacher-gen-math.jsonl", encoding="utf-8")) if trace_id(r) in keep]
    picked = random.Random(args.seed).sample(rows, args.count)
    with open(C / "nothink-pilot-problems.jsonl", "w", encoding="utf-8") as out:
        for r in picked:
            out.write(json.dumps({"id": r["id"], "problem": r["problem"], "answer": r["reference"],
                                  "trace": trace_id(r)}) + "\n")
    print("%d problems -> %s" % (len(picked), C / "nothink-pilot-problems.jsonl"))


def render(tok, problem):
    from merge_proxy import PROMPT

    return tok.apply_chat_template([{"role": "user", "content": PROMPT.format(problem=problem)}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False, reasoning_effort="medium")


def inputs(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TEACHER)
    end = tok.convert_tokens_to_ids("<|im_end|>")
    problems = [json.loads(line) for line in open(C / "nothink-pilot-problems.jsonl", encoding="utf-8")]
    traces = {trace_id(r): r for r in map(json.loads, open(C / "teacher-gen-math.jsonl", encoding="utf-8"))}
    native = {r["id"]: r for r in map(json.loads, open(C / "nothink-pilot-native.jsonl", encoding="utf-8"))}
    starts, n = {}, 0
    with open(C / "nothink-pilot.jsonl", "w", encoding="utf-8") as out:
        for p in problems:
            prompt = tok(render(tok, p["problem"]), add_special_tokens=False)["input_ids"]
            answers = {"post": traces[p["trace"]]["text"].split("</think>")[-1].strip()}
            if p["id"] in native and native[p["id"]]["finished"]:
                answers["native"] = native[p["id"]]["text"].strip()
            for kind, text in answers.items():
                doc = "%s:%s" % (kind, p["id"])
                ids = prompt + tok(text, add_special_tokens=False)["input_ids"] + [end]
                out.write(json.dumps({"doc_id": doc, "split": "train", "input_ids": ids}) + "\n")
                starts[doc] = len(prompt)
                n += 1
    (C / "nothink-pilot-starts.json").write_text(json.dumps(starts), encoding="utf-8")
    graded = [native[p["id"]] for p in problems if p["id"] in native]
    print("%d documents -> %s; native non-thinking: %d answered, %d correct, %d finished"
          % (n, C / "nothink-pilot.jsonl", len(graded), sum(r["correct"] for r in graded), sum(r["finished"] for r in graded)))


def score(args):
    from teacher_kl import CachedTeacher

    starts = json.loads((C / "nothink-pilot-starts.json").read_text(encoding="utf-8"))
    teacher = CachedTeacher(args.cache, "train", device="cpu")
    stats = {"post": [], "native": []}
    for doc_id in teacher.ids:
        kind = doc_id.split(":", 1)[0]
        r = teacher.cache.read_document(doc_id)
        ids, top, lp = np.asarray(r["input_ids"]), r["topk_ids"], r["topk_logprobs"].astype(np.float32)
        at = np.arange(starts[doc_id] - 1, len(ids) - 1)  # positions predicting the answer and its close
        hit = top[at] == ids[at + 1][:, None]
        # Outside the top-64 the teacher's probability is below its 64th; that bound stands in.
        nll = np.where(hit.any(1), -np.where(hit, lp[at], -np.inf).max(1), -lp[at, -1])
        stats[kind].append({"doc": doc_id, "tokens": len(at), "nll": float(nll.mean()),
                            "outside": float((~hit.any(1)).mean()), "surprises": int((nll > np.log(100)).sum()),
                            "worst": float(nll.max())})
    for kind, rows in stats.items():
        if not rows:
            continue
        print("%-6s answers %3d  tokens median %4d  NLL/token median %.3f mean %.3f  outside top-64 %.2f%%  "
              "tokens with p<0.01 per answer median %d p90 %d  worst-token NLL median %.1f"
              % (kind, len(rows), statistics.median(r["tokens"] for r in rows), statistics.median(r["nll"] for r in rows),
                 statistics.mean(r["nll"] for r in rows), 100 * statistics.mean(r["outside"] for r in rows),
                 statistics.median(r["surprises"] for r in rows),
                 sorted(r["surprises"] for r in rows)[int(0.9 * len(rows))], statistics.median(r["worst"] for r in rows)))
    (C / "nothink-pilot-scores.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--count", type=int, default=200)
    b.add_argument("--seed", type=int, default=5)
    sub.add_parser("inputs")
    s = sub.add_parser("score")
    s.add_argument("--cache", type=Path, default=C.parent / "teacher-cache-nothink-pilot")
    args = parser.parse_args()
    {"build": build, "inputs": inputs, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
