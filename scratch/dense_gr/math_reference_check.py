"""Check the replay corpora's math traces against their source datasets' reference answers.

The traces are frontier-model answers (Qwen3.8-Max, GLM-5.2, Kimi K3) to prompts taken
from public math sets; the distillation dataset keeps no reference answer or verdict, so
this goes back to the sources, matches each trace's question, and compares final answers
with math_verify: GSM8K and MATH (train), MetaMathQA ("The answer is: X"), Orca-Math (the
last number of its solution), NuminaMath-1.5 (its curated `answer`, skipping proofs and
problems it marks invalid). Sources are streamed and only matched questions kept.

A mismatch is a candidate error on either side -- reference sets have their own -- so the
output keeps both answers for review.

    python scratch/dense_gr/math_reference_check.py --corpora ../capture-data/thinking-code-math.jsonl \\
        ../capture-data/think-first-5m.jsonl --output ../capture-data/math-reference-check.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "downstream" / "math_bench"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rollout_prompts import SUBJECTS  # noqa: E402
from run_math import boxed, correct  # noqa: E402

USER = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)


def norm(text):
    return re.sub(r"\s+", " ", text or "").strip().lower()


def trace_answer(text):
    reply = text.split("<|im_start|>assistant")[-1].split("</think>")[-1]
    found = boxed(reply)
    if found is None:
        match = re.findall(r"answer is:?\s*\$?([^\n$]+)", reply)
        found = match[-1].strip().rstrip(".") if match else None
    return found


def last_number(text):
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text or "")
    return numbers[-1].replace(",", "").rstrip(".") if numbers else None


def references(wanted):
    """{normalized question: reference answer} for the questions in `wanted`."""
    from datasets import load_dataset

    found = {}

    def take(question, answer):
        key = norm(question)
        if key in wanted and answer not in (None, "") and key not in found:
            found[key] = answer

    for r in load_dataset("openai/gsm8k", "main", split="train"):
        take(r["question"], r["answer"].split("####")[-1].strip().replace(",", ""))
    for subject in SUBJECTS:
        for r in load_dataset("EleutherAI/hendrycks_math", subject, split="train"):
            take(r["problem"], boxed(r["solution"]))
    for r in load_dataset("meta-math/MetaMathQA", split="train", streaming=True):
        match = re.findall(r"The answer is:?\s*(.+)$", r["response"].strip())
        take(r["query"], match[-1].strip() if match else None)
    for r in load_dataset("microsoft/orca-math-word-problems-200k", split="train", streaming=True):
        take(r["question"], last_number(r["answer"]))
    for r in load_dataset("AI-MO/NuminaMath-1.5", split="train", streaming=True):
        if r.get("problem_is_valid") == "Yes" and r.get("solution_is_valid") != "No" \
                and str(r.get("answer", "")).lower() not in ("proof", "notfound", ""):
            take(r["problem"], r["answer"])
        if len(found) == len(wanted):
            break
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpora", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    traces = []
    for path in args.corpora:
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            if row.get("domain") != "math":
                continue
            users = USER.findall(row["text"])
            if users:
                traces.append((path.stem, row["doc_id"], str(row.get("source")), norm(users[-1]),
                               trace_answer(row["text"])))
    print("%d math traces" % len(traces), flush=True)
    refs = references({question for _, _, _, question, _ in traces})
    counts = {}
    with open(args.output, "w", encoding="utf-8") as out:
        for corpus, doc_id, source, question, answer in traces:
            reference = refs.get(question)
            if reference is None:
                verdict = "unmatched"
            elif answer is None:
                verdict = "no_answer"
            else:
                verdict = "agrees" if correct(answer, str(reference)) else "disagrees"
            c = counts.setdefault(source.split("/")[0], {})
            c[verdict] = c.get(verdict, 0) + 1
            out.write(json.dumps({"doc_id": doc_id, "corpus": corpus, "source": source, "verdict": verdict,
                                  "answer": answer, "reference": reference}, ensure_ascii=False) + "\n")
    print("%-16s %8s %9s %10s %9s %10s" % ("source", "matched", "agrees", "disagrees", "no answer", "unmatched"))
    for source, c in sorted(counts.items(), key=lambda kv: -sum(kv[1].values())):
        matched = c.get("agrees", 0) + c.get("disagrees", 0) + c.get("no_answer", 0)
        print("%-16s %8d %8.1f%% %9.1f%% %9d %10d" % (
            source, matched, 100 * c.get("agrees", 0) / max(matched, 1),
            100 * c.get("disagrees", 0) / max(matched, 1), c.get("no_answer", 0), c.get("unmatched", 0)))


if __name__ == "__main__":
    main()
