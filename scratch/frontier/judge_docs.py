"""A frontier judge over replay documents nothing else verifies.

The math traces were checked against reference answers and the code for parsing; the
reasoning and multiple-choice documents (think-first, the non-math part of thinking) were
never checked at all. Each document's conversation goes to the judge, which says whether
the final answer is correct and how sound the reasoning is; `collect` turns confident
"incorrect" verdicts, and truncated or garbled responses, into an exclusion list for
`--exclude-documents`.

    python scratch/frontier/judge_docs.py build --inputs ../capture-data/recapture-think-first.jsonl ... \\
        --output ../capture-data/frontier/judge-requests.jsonl
    python scratch/frontier/judge_docs.py collect --responses ../capture-data/frontier/judge-responses.jsonl \\
        --output ../capture-data/exclude-judged.json
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

TOKENIZER = "D:/DeepThought/Projects/HybridModel/DistillKit/scratch/dense_gr/merges-r6r8b/u50"

INSTRUCTIONS = """You audit training data for a small language model. Below is one training conversation \
(system and user turns, then the assistant's response, which may include reasoning in <think> tags). Judge \
the assistant's final answer and reasoning on their merits; solve or check the problem yourself.

Return only JSON: {"final_correct": "yes" | "no" | "unsure", "confidence": 0-1, "reasoning_sound": 1-5, \
"issues": [short strings: e.g. "wrong final answer", "arithmetic error", "hallucinated fact", "truncated", \
"answer contradicts reasoning", "format violated", "garbled text"]}"""


def build(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for path in args.inputs:
            for line in open(path, encoding="utf-8"):
                row = json.loads(line)
                if args.skip and re.search(args.skip, row["doc_id"]):
                    continue
                text = tokenizer.decode(row["input_ids"])
                if "<|im_start|>assistant" not in text:
                    continue
                out.write(json.dumps({"id": row["doc_id"], "messages": [
                    {"role": "system", "content": INSTRUCTIONS},
                    {"role": "user", "content": "<conversation>\n%s\n</conversation>" % text}],
                    "max_tokens": 6000, "reasoning": {"effort": "medium"},
                    "response_format": {"type": "json_object"}}) + "\n")
                count += 1
    print("%d requests -> %s" % (count, args.output))


def collect(args):
    verdicts, excluded = Counter(), []
    for row in map(json.loads, open(args.responses, encoding="utf-8")):
        try:
            verdict = json.loads(re.sub(r"^```(json)?|```$", "", (row.get("content") or "").strip()))
        except ValueError:
            verdicts["unparsable"] += 1
            continue
        issues = " ".join(map(str, verdict.get("issues") or [])).lower()
        wrong = verdict.get("final_correct") == "no" and float(verdict.get("confidence") or 0) >= args.confidence
        broken = any(word in issues for word in ("truncated", "garbled"))
        verdicts["final " + str(verdict.get("final_correct"))] += 1
        if wrong or broken:
            excluded.append(row["id"])
            verdicts["excluded"] += 1
    args.output.write_text(json.dumps(sorted(excluded), indent=0), encoding="utf-8")
    print("verdicts: %s\n%d excluded -> %s" % (dict(verdicts.most_common()), len(excluded), args.output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--inputs", type=Path, nargs="+", required=True)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--skip", default=None, help="regex of doc ids to leave out (already verified)")
    c = sub.add_parser("collect")
    c.add_argument("--responses", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--confidence", type=float, default=0.8)
    args = parser.parse_args()
    (build if args.command == "build" else collect)(args)


if __name__ == "__main__":
    main()
