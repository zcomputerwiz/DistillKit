"""Requests for long-context questions over each long document (frontier_batch.py input).

Eight questions a document, spread across it, each with a short reference answer and
verbatim evidence quotes so `qa_verify.py` can check them against the text without
another model: three find a specific detail, two connect facts from files far apart, one
counts or lists, one traces what code does, and one asks something the document does
not say -- so the model learns to say so rather than invent.

    python scratch/frontier/qa_requests.py --docs ../capture-data/long-docs-code.jsonl \\
        --output ../capture-data/frontier/qa-code-requests.jsonl
"""
import argparse
import json
from pathlib import Path

INSTRUCTIONS = """You write reading-comprehension questions that test whether a reader used a long document, \
for training a small model to work with long contexts. The document below is a dump of source files from one \
repository, each starting with a line "# ===== File: <path> =====".

Write exactly 8 questions:
- 3 of type "find": a specific detail stated in one file (a constant's value, a default argument, an error \
message, a condition), phrased so it cannot be answered from general knowledge of the library.
- 2 of type "connect": the answer needs facts from two different files that are far apart in the document.
- 1 of type "count": count or list items with some property (e.g. the functions in a file that raise ValueError).
- 1 of type "trace": what a specific function returns or does for a specific concrete input, worked out from \
its code here.
- 1 of type "absent": a plausible question about this code whose answer is NOT in the document; its answer \
says the document does not contain it.

Spread the questions over the whole document: at least two about its first third, two about its middle, and \
two about its last third. Answers are short and exact (one or two sentences, or a value). Evidence quotes \
must be copied character for character from the document, each at most 200 characters, with the file path \
they come from; "absent" questions have no evidence.

Return only JSON: {"items": [{"type": str, "question": str, "answer": str, \
"evidence": [{"file": str, "quote": str}]}]}"""


MULTI = INSTRUCTIONS.replace("The document below is a dump", "Each document below (in <document id=...> tags) is a dump").replace(
    "Write exactly 8 questions:", "For EACH document separately, write exactly 8 questions about that document alone:").replace(
    'Return only JSON: {"items": [{', 'Return only JSON: {"documents": [{"doc_id": the document id, "items": [{').replace(
    '"evidence": [{"file": str, "quote": str}]}]}', '"evidence": [{"file": str, "quote": str}]}]}]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--per-request", type=int, default=1)
    parser.add_argument("--done", type=Path, default=None, help="earlier responses: skip documents they answered")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.done:
        done = {r["id"] for r in map(json.loads, open(args.done, encoding="utf-8")) if "error" not in r}
    docs = [d for d in map(json.loads, open(args.docs, encoding="utf-8")) if d["doc_id"] not in done]
    with open(args.output, "w", encoding="utf-8") as out:
        for start in range(0, len(docs), args.per_request):
            group = docs[start:start + args.per_request]
            if len(group) == 1:
                system, body, request_id = INSTRUCTIONS, "<document>\n%s</document>" % group[0]["text"], group[0]["doc_id"]
            else:
                # Several documents a request: the free tier allows 1,000 requests a day.
                system = MULTI
                body = "\n\n".join('<document id="%s">\n%s</document>' % (d["doc_id"], d["text"]) for d in group)
                request_id = "qa-batch:" + "|".join(d["doc_id"] for d in group)
            out.write(json.dumps({
                "id": request_id,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": body}],
                "max_tokens": 12000 * len(group), "reasoning": {"effort": args.effort},
                "response_format": {"type": "json_object"}}) + "\n")
    print("%d documents (%d done) in %d requests -> %s"
          % (len(docs), len(done), -(-len(docs) // args.per_request), args.output))


if __name__ == "__main__":
    main()
