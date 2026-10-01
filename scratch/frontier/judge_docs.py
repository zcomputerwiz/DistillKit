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

"final_correct" is "no" only when the main answer is wrong: the wrong choice, a wrong result, code that fails \
on ordinary inputs, a false factual claim the answer rests on. Unhandled extreme edge cases (overflow at \
INT_MAX, NaN input), portability, style, efficiency or missing caveats are "yes", with the issue listed and \
severity "minor". Tool calls in this data are written inline as <tool_call> XML blocks: that is the correct \
format, not a violation.

Return only JSON: {"final_correct": "yes" | "no" | "unsure", "severity": "none" | "minor" | "major", \
"confidence": 0-1, "reasoning_sound": 1-5, "issues": [short strings: e.g. "wrong final answer", "arithmetic \
error", "hallucinated fact", "truncated", "answer contradicts reasoning", "garbled text"]}"""


BATCHED = INSTRUCTIONS.replace("Below is one training conversation", "Below are several training conversations, "
                                "each in <conversation id=...> tags, judged independently;").replace(
    'Return only JSON: {', 'Return only JSON: {"verdicts": [one per conversation: {"id": the conversation id, ') + "]}"


def judged(paths):
    """Document ids already given a verdict, from single or batched responses."""
    done = set()
    for path in paths or []:
        for row in map(json.loads, open(path, encoding="utf-8")):
            done |= {doc_id for doc_id, _ in verdicts_of(row)}
    return done


def build(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    done = judged(args.done)
    docs = []
    for path in args.inputs:
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            if (args.skip and re.search(args.skip, row["doc_id"])) or row["doc_id"] in done:
                continue
            text = tokenizer.decode(row["input_ids"])
            if "<|im_start|>assistant" in text:
                docs.append((row["doc_id"], text))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Several conversations a request: the free tier allows 1,000 requests a day.
    with open(args.output, "w", encoding="utf-8") as out:
        for start in range(0, len(docs), args.per_request):
            group = docs[start:start + args.per_request]
            body = "\n\n".join('<conversation id="%s">\n%s\n</conversation>' % pair for pair in group)
            out.write(json.dumps({"id": "judge-batch:%d" % start, "messages": [
                {"role": "system", "content": BATCHED if len(group) > 1 else INSTRUCTIONS},
                {"role": "user", "content": body}],
                "max_tokens": 4000 + 1500 * len(group), "reasoning": {"effort": "medium"},
                "response_format": {"type": "json_object"}}) + "\n")
    print("%d documents (%d already judged) in %d requests -> %s"
          % (len(docs), len(done), -(-len(docs) // args.per_request), args.output))


def verdicts_of(row):
    """(doc id, verdict) pairs from one response, single or batched."""
    if "error" in row:
        return []
    try:
        reply = json.loads(re.sub(r"^```(json)?|```$", "", (row.get("content") or "").strip()))
    except ValueError:
        return []
    if not isinstance(reply, dict):
        return []
    if "verdicts" in reply:
        return [(str(v.get("id")), v) for v in reply["verdicts"] if isinstance(v, dict) and v.get("id")]
    return [(row["id"], reply)]


def collect(args):
    verdicts, excluded, seen = Counter(), [], set()
    for path in args.responses:
        for row in map(json.loads, open(path, encoding="utf-8")):
            for doc_id, verdict in verdicts_of(row):
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                issues = " ".join(map(str, verdict.get("issues") or [])).lower()
                try:
                    confidence = float(verdict.get("confidence") or 0)
                except (TypeError, ValueError):
                    confidence = 0.0
                # The first 598 verdicts predate "severity"; they are cross-checked by a second judge.
                wrong = (verdict.get("final_correct") == "no" and confidence >= args.confidence
                         and verdict.get("severity", "major") == "major")
                broken = any(word in issues for word in ("truncated", "garbled"))
                verdicts["final " + str(verdict.get("final_correct"))] += 1
                if wrong or broken:
                    excluded.append({"id": doc_id, "issues": verdict.get("issues")})
                    verdicts["excluded"] += 1
    args.output.write_text(json.dumps(sorted(e["id"] for e in excluded), indent=0), encoding="utf-8")
    args.output.with_suffix(".detail.jsonl").write_text("".join(json.dumps(e) + "\n" for e in excluded),
                                                       encoding="utf-8")
    print("%d judged; verdicts: %s\n%d excluded -> %s"
          % (len(seen), dict(verdicts.most_common()), len(excluded), args.output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--inputs", type=Path, nargs="+", required=True)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--skip", default=None, help="regex of doc ids to leave out (already verified)")
    b.add_argument("--done", type=Path, nargs="*", default=None, help="earlier responses: skip what they judged")
    b.add_argument("--per-request", type=int, default=10)
    c = sub.add_parser("collect")
    c.add_argument("--responses", type=Path, nargs="+", required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--confidence", type=float, default=0.8)
    args = parser.parse_args()
    (build if args.command == "build" else collect)(args)


if __name__ == "__main__":
    main()
