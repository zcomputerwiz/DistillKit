"""Check frontier long-context questions against their documents, keep what holds up.

A question is kept when its answer is non-empty and short, and every evidence quote occurs
in its document -- whitespace-normalized, inside the file it names -- or, for "absent"
questions, when it carries no evidence. Prints per-type yields and writes the kept items
per document.

    python scratch/frontier/qa_verify.py --docs ../capture-data/long-docs-code.jsonl \\
        --responses ../capture-data/frontier/qa-code-responses.jsonl --output ../capture-data/frontier/qa-code.jsonl
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

HEADER = re.compile(r"^# ===== File: (.+?) =====$", re.M)
SPACE = re.compile(r"\s+")


def files_of(text):
    """{path: normalized body} for a document of headed files."""
    marks = list(HEADER.finditer(text))
    return {m.group(1): SPACE.sub(" ", text[m.end():marks[i + 1].start() if i + 1 < len(marks) else len(text)])
            for i, m in enumerate(marks)}


def check(item, files, whole):
    if not isinstance(item, dict) or not str(item.get("question", "")).strip():
        return "malformed"
    answer = str(item.get("answer", "")).strip()
    if not answer or len(answer) > 600:
        return "answer missing or long"
    evidence = item.get("evidence") or []
    if item.get("type") == "absent":
        return "ok" if not evidence else "absent with evidence"
    if not evidence:
        return "no evidence"
    for quote in evidence:
        text = SPACE.sub(" ", str(quote.get("quote", ""))).strip()
        if len(text) < 8:
            return "quote too short"
        body = files.get(str(quote.get("file", "")).strip())
        if body is None:
            # Paths are sometimes given without the source prefix; accept a unique suffix.
            matches = [b for p, b in files.items() if p.endswith(str(quote.get("file", "")).strip())]
            body = matches[0] if len(matches) == 1 else None
        if body is None:
            if text not in whole:
                return "file unknown and quote not found"
        elif text not in body:
            return "quote not in file"
    return "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--responses", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    docs = {d["doc_id"]: d for d in map(json.loads, open(args.docs, encoding="utf-8"))}
    outcomes, kept_types, kept_docs = Counter(), Counter(), 0
    with open(args.output, "w", encoding="utf-8") as out:
        for row in (json.loads(l) for path in args.responses for l in open(path, encoding="utf-8")):
            if "error" in row:
                outcomes["request failed"] += 1
                continue
            try:
                reply = json.loads(re.sub(r"^```(json)?|```$", "", row["content"].strip()))
                # One document a request ({"items"}) or several ({"documents": [{"doc_id", "items"}]}).
                groups = ([(row["id"], reply["items"])] if "items" in reply
                          else [(d["doc_id"], d["items"]) for d in reply["documents"]])
            except (ValueError, KeyError, TypeError, AttributeError):
                outcomes["unparsable response"] += 1
                continue
            for doc_id, items in groups:
                if doc_id not in docs:
                    outcomes["unknown document"] += 1
                    continue
                kept_docs += write_kept(out, doc_id, items, docs[doc_id]["text"], outcomes, kept_types)
    print("outcomes: %s" % dict(outcomes.most_common()))
    print("kept %d questions over %d documents: %s" % (sum(kept_types.values()), kept_docs, dict(kept_types)))


def write_kept(out, doc_id, items, text, outcomes, kept_types):
    """Check one document's items, write the ones that hold; 1 if any did."""
    files, whole = files_of(text), SPACE.sub(" ", text)
    kept = []
    for item in items if isinstance(items, list) else []:
        verdict = check(item, files, whole)
        outcomes[verdict] += 1
        if verdict == "ok":
            kept.append({k: item[k] for k in ("type", "question", "answer", "evidence") if k in item})
            kept_types[item.get("type")] += 1
    if kept:
        out.write(json.dumps({"doc_id": doc_id, "items": kept}) + "\n")
    return int(bool(kept))

if __name__ == "__main__":
    main()
