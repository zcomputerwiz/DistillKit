"""Collect thinking-mode documents in the native format, sized to be trained on whole.

The students trained almost entirely on assistant turns that open on an empty, closed
think block, and in thinking mode they reason for ~1,200 tokens and loop. The cause of that
format is here, in rendering: the distillation dataset puts `<think>...</think>` inside
the assistant `content`, and the teacher's template then prepends its own empty block to
any message without `reasoning_content`. So this renders natively -- the reasoning moved to
`reasoning_content`, where the template places it itself -- and draws on the dataset's
code, math and reasoning configs, of which the 5M corpus used none.

Every document must fit the training cap whole (`--max-tokens`, 1024), so the model always
sees its reasoning conclude and the answer follow: a prefix cap that cuts reasoning off
teaches a model to keep going. Reasoning is de-hedged by `dehedge.py`'s rules; documents
are screened against the scored benchmarks as `expand_corpus.py` screens, deduplicated,
and skipped if the 5M corpus already has them. Every 20th kept document is held out.

    python scratch/dense_gr/thinking_corpus.py --output ../capture-data/thinking-code-math.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dehedge import edit_block  # noqa: E402
from expand_corpus import build_banks, fingerprint, shingles  # noqa: E402

REPO = "r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation"
TEACHER = "D:/DeepThought/Projects/HybridModel/teacher-hf"
THINK = re.compile(r"^\s*<think>(.*?)</think>\s*", re.S)


def rows(config):
    """Train rows of one config, streamed from its parquet shards."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    shards = sorted(f for f in fs.ls("datasets/%s/data/%s" % (REPO, config), detail=False)
                    if f.endswith(".parquet") and "/train-" in f.replace("\\", "/"))
    for shard in shards:
        with fs.open(shard, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            for group in range(parquet.num_row_groups):
                yield from parquet.read_row_group(group).to_pylist()


def native(messages):
    """Messages with each assistant turn's leading think block moved to reasoning_content,
    de-hedged; and the total reasoning length in characters."""
    out, reasoning = [], 0
    for message in messages:
        role, content = message.get("role"), message.get("content") or ""
        if not role or not content.strip():
            continue
        entry = {"role": role, "content": content}
        if role == "assistant":
            match = THINK.match(content)
            if match:
                thought = edit_block(match.group(1))[0].strip()
                entry = {"role": role, "content": content[match.end():],
                         "reasoning_content": thought}
                reasoning += len(thought)
        out.append(entry)
    return out, reasoning


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", default=["sft_code", "sft_math", "sft_reasoning"])
    parser.add_argument("--tokens", type=int, default=3_000_000)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--min-reasoning", type=int, default=64, help="tokens")
    parser.add_argument("--existing", type=Path, nargs="*",
                        default=[Path("../capture-data/run5m.jsonl")])
    parser.add_argument("--eval-every", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TEACHER)
    scored, related = build_banks()
    seen_ids, seen_prints = set(), set()
    for path in args.existing:
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            seen_ids.add(str(row["doc_id"]).split(":")[0])
            seen_prints.add(fingerprint(row["text"]))
    stats = {c: dict(kept=0, tokens=0, known=0, long=0, thin=0, scored=0, related=0,
                     duplicate=0, empty=0) for c in args.configs}
    streams = {c: rows(c) for c in args.configs}
    total = kept = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for config in itertools.cycle(list(streams)):
            if total >= args.tokens or not streams:
                break
            if config not in streams:
                continue
            row = next(streams[config], None)
            if row is None:
                del streams[config]
                continue
            s = stats[config]
            ident = str(row.get("id") or row.get("parent_id") or "")
            if ident in seen_ids:
                s["known"] += 1
                continue
            messages, reasoning = native(row.get("messages") or [])
            if not messages or not any(m["role"] == "assistant" for m in messages):
                s["empty"] += 1
                continue
            text = tok.apply_chat_template(messages, tokenize=False)
            length = len(tok(text, add_special_tokens=False)["input_ids"])
            if length > args.max_tokens:
                s["long"] += 1
                continue
            thought = sum(len(tok(m.get("reasoning_content", ""), add_special_tokens=False)["input_ids"])
                          for m in messages)
            if thought < args.min_reasoning:
                s["thin"] += 1
                continue
            grams = shingles(text)
            if grams & scored:
                s["scored"] += 1
                continue
            if grams & related:
                s["related"] += 1
                continue
            key = fingerprint(text)
            if key in seen_prints:
                s["duplicate"] += 1
                continue
            seen_prints.add(key)
            split = "eval" if kept % args.eval_every == 0 else "train"
            out.write(json.dumps({"doc_id": "think:%s:%s" % (config, ident or kept), "text": text,
                                  "split": split, "source": row.get("source") or config,
                                  "domain": row.get("domain") or config, "tokens": length,
                                  "reasoning_tokens": thought}, ensure_ascii=False) + "\n")
            kept += 1
            s["kept"] += 1
            s["tokens"] += length
            total += length
            if kept % 500 == 0:
                print("kept %d documents, %d tokens" % (kept, total), flush=True)
    for config, s in stats.items():
        print("%-14s %s" % (config, json.dumps(s)))
    print("wrote %d documents, %d tokens -> %s" % (kept, total, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
