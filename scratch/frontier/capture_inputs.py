"""The verified frontier data as capture input, rendered with the served chat template.

qa: each long code document with its checked questions as a conversation -- the document
and first question in the first user turn, then answer, question, answer -- in order
through the document. Questions that would push it past --max-length are left off. All
tokens are meant to be scored (the document is long-context language modelling too).

tools: each verified tool conversation with its tool list; meant for --assistant-only-caches.

Non-thinking turns (the answers are short and carry no reasoning). A tenth of each set,
chosen by a hash of its id, is held out.

    python scratch/frontier/capture_inputs.py qa --output ../capture-data/frontier-qa.jsonl
    python scratch/frontier/capture_inputs.py tools --output ../capture-data/frontier-tools.jsonl
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dense_gr"))
from long_context_probe import TOKENIZER  # noqa: E402

F = Path(__file__).resolve().parents[3] / "capture-data"


def split_of(key, share=0.1):
    return "eval" if int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share else "train"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["qa", "tools"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=32768)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    def ids_of(messages, tools=None):
        text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, enable_thinking=False)
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    rows, dropped, tokens = [], 0, 0
    if args.kind == "qa":
        docs = {d["doc_id"]: d["text"] for d in map(json.loads, open(F / "long-docs-code.jsonl", encoding="utf-8"))}
        for row in map(json.loads, open(F / "frontier" / "qa-code.jsonl", encoding="utf-8")):
            messages, ids = [], None
            for n, item in enumerate(row["items"]):
                question = item["question"] if n else "<document>\n%s</document>\n\n%s" % (docs[row["doc_id"]], item["question"])
                trial = ids_of(messages + [{"role": "user", "content": question},
                                           {"role": "assistant", "content": item["answer"]}])
                if len(trial) > args.max_length:
                    dropped += len(row["items"]) - n
                    break
                messages += [{"role": "user", "content": question}, {"role": "assistant", "content": item["answer"]}]
                ids = trial
            if ids:
                rows.append({"doc_id": "qa:" + row["doc_id"], "split": split_of(row["doc_id"]), "input_ids": ids,
                             "questions": len(messages) // 2})
    else:
        for name in ("tools.jsonl", "tools2.jsonl"):
            for row in map(json.loads, open(F / "frontier" / name, encoding="utf-8")):
                # Tool results sometimes come back as JSON objects; the template wants text.
                messages = [{**m, "content": m.get("content") if isinstance(m.get("content"), str)
                             else "" if m.get("content") is None else json.dumps(m["content"])}
                            for m in row["messages"]]
                # The template drops the text of an assistant message followed by another
                # assistant message; the model writes such a pair as one turn.
                merged = []
                for m in messages:
                    if merged and m["role"] == merged[-1]["role"] == "assistant":
                        last = merged[-1]
                        last["content"] = "\n\n".join(c for c in (last["content"].strip(), m["content"].strip()) if c)
                        last["tool_calls"] = (last.get("tool_calls") or []) + (m.get("tool_calls") or [])
                    else:
                        merged.append(dict(m))
                ids = ids_of(merged, row["tools"])
                if len(ids) > args.max_length:
                    dropped += 1
                    continue
                rows.append({"doc_id": "tools:" + row["doc_id"], "split": split_of(row["doc_id"]), "input_ids": ids,
                             "scenario": row["scenario"]})
    with open(args.output, "w", encoding="utf-8") as out:
        for row in rows:
            tokens += len(row["input_ids"])
            out.write(json.dumps(row) + "\n")
    print("%d documents, %d tokens (max %d), %d eval; %d %s left off -> %s"
          % (len(rows), tokens, max(len(r["input_ids"]) for r in rows), sum(r["split"] == "eval" for r in rows),
             dropped, "questions" if args.kind == "qa" else "conversations", args.output))


if __name__ == "__main__":
    main()
