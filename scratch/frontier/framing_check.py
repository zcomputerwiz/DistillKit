"""Where should a long code document sit for the teacher to predict it as code?

In a user turn the chat teacher's top-1 is a false `<|im_end|>` at 35-49% of positions
(it expects the person to stop typing at every line break); in raw text it is 0% and
calibrated. Round 2 distilled 15M tokens of code from the user-turn view and its long-code
NLL got worse with distance. This renders the same documents three ways -- a user turn, a
`read_file` tool result, raw text -- for one teacher capture, then scores the teacher on
the document's own tokens in each: top-1 accuracy, mean top-1 probability, false end-of-turn.

    python scratch/frontier/framing_check.py build --output ../capture-data/framing-check.jsonl
    (capture it with distillkit.sample_transformers at 32K)
    python scratch/frontier/framing_check.py score --cache ../teacher-cache-framing-check
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dense_gr"))
from capture_inputs import F, chat_ids  # noqa: E402

READ_FILE = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a file from the repository.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
BUCKETS = [(0, 2048), (2048, 8192), (8192, 1 << 30)]


def renderings(tokenizer, text):
    yield "user", chat_ids(tokenizer, [{"role": "user", "content": "<document>\n%s</document>" % text}])
    yield "tool", chat_ids(tokenizer, [
        {"role": "user", "content": "Read the repository dump in repo.txt."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "read_file", "arguments": {"path": "repo.txt"}}}]},
        {"role": "tool", "content": text}], tools=READ_FILE)
    yield "raw", tokenizer(text, add_special_tokens=False)["input_ids"]


def build(args):
    from transformers import AutoTokenizer
    from long_context_probe import TOKENIZER

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    docs = [d for d in map(json.loads, open(args.docs, encoding="utf-8")) if d["tokens"] >= args.min_tokens]
    picked, seen = [], set()
    for d in docs:  # one document per source library, for spread
        if d["source"] not in seen:
            seen.add(d["source"])
            picked.append(d)
        if len(picked) == args.count:
            break
    with open(args.output, "w", encoding="utf-8") as out:
        for d in picked:
            for framing, ids in renderings(tokenizer, d["text"]):
                out.write(json.dumps({"doc_id": "%s|%s" % (framing, d["doc_id"]), "split": "train",
                                      "input_ids": ids[:args.max_length]}) + "\n")
    print("%d documents x 3 framings -> %s" % (len(picked), args.output))


def document_region(ids, framing, c):
    """Positions whose next token is the document's own text."""
    region = np.zeros(len(ids), dtype=bool)
    if framing == "raw":
        region[:] = True
    else:
        if framing == "tool":  # the last tag: the system prompt's tool instructions may name it
            closer = c("</tool_response>")
            start = int(np.nonzero(ids == c("<|im_start|>"))[0][-1])
            start = int(np.nonzero(ids[start:] == c("<tool_response>"))[0][0]) + start + 1
        else:  # after `<|im_start|>user`
            closer = c("<|im_end|>")
            starts = np.nonzero(ids[:-1] == c("<|im_start|>"))[0]
            start = int([i for i in starts if ids[i + 1] == c("user")][0]) + 2
        stops = np.nonzero(ids[start:] == closer)[0]
        region[start:start + int(stops[0]) if len(stops) else len(ids)] = True
    target = np.zeros(len(ids), dtype=bool)
    target[:-1] = region[1:]
    target[-2:] = False  # the last kept target predicts the truncation, not text
    return target


def score(args):
    from transformers import AutoTokenizer

    from long_context_probe import TOKENIZER
    from teacher_kl import CachedTeacher

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    c = tokenizer.convert_tokens_to_ids
    end = c("<|im_end|>")
    teacher = CachedTeacher(args.cache, "train", device="cpu")
    stats = {}
    for doc_id in teacher.ids:
        framing = doc_id.split("|", 1)[0]
        r = teacher.cache.read_document(doc_id)
        ids, top = np.asarray(r["input_ids"]), r["topk_ids"]
        p = np.exp(r["topk_logprobs"][:, 0].astype(np.float32))
        target = document_region(ids, framing, c)
        where = np.nonzero(target)[0]
        offset = where - where[0]
        hit = top[where, 0] == ids[where + 1]
        false_end = (top[where, 0] == end) & (ids[where + 1] != end)
        for bucket in [None] + BUCKETS:
            keep = np.ones(len(where), bool) if bucket is None else (offset >= bucket[0]) & (offset < bucket[1])
            s = stats.setdefault((framing, bucket), [0, 0.0, 0.0, 0])
            s[0] += int(keep.sum()); s[1] += float(hit[keep].sum()); s[2] += float(p[where][keep].sum())
            s[3] += int(false_end[keep].sum())
    result = {}
    for (framing, bucket), (n, hits, conf, ends) in sorted(stats.items(), key=lambda kv: (kv[0][0], kv[0][1] or (-1,))):
        label = "all" if bucket is None else "%dk+" % (bucket[0] // 1024)
        row = dict(tokens=n, accuracy=hits / max(n, 1), confidence=conf / max(n, 1), false_end=ends / max(n, 1))
        result.setdefault(framing, {})[label] = row
        print("%-5s %-5s tokens %8d  top-1 %.3f  confidence %.3f  false end-of-turn %.3f"
              % (framing, label, n, row["accuracy"], row["confidence"], row["false_end"]))
    if args.output:
        args.output.write_text(json.dumps(result, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--docs", type=Path, default=F / "long-docs-code.jsonl")
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--count", type=int, default=12)
    b.add_argument("--min-tokens", type=int, default=16384)
    b.add_argument("--max-length", type=int, default=32768)
    s = sub.add_parser("score")
    s.add_argument("--cache", type=Path, required=True)
    s.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    build(args) if args.command == "build" else score(args)


if __name__ == "__main__":
    main()
