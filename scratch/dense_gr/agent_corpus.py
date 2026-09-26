"""Render agent and tool-use traces with their tool calls, in the native thinking format.

`prepare_corpus.render_messages` kept each message's role and content and nothing else, so
every agent trace in the 5M corpus lost its `tool_calls` and its tool definitions: an
assistant turn announces "Let me start by inspecting the file" and ends, and a tool
response follows from nowhere. All 514 agent documents were trained that way -- teaching,
for an agentic model, exactly the wrong thing.

This renders them whole: the row's `tools` (their schemas un-stringified from
`parameters_json`) go to the template, each assistant turn keeps its `tool_calls` (JSON
arguments parsed) and has its leading think block moved to `reasoning_content`, de-hedged,
and tool results keep their names. The teacher's template does the rest, in Qwen3.5's own
`<tool_call><function=...>` format.

    python scratch/dense_gr/agent_corpus.py --output ../capture-data/agent-tools.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dehedge import edit_block  # noqa: E402
from expand_corpus import build_banks, fingerprint, shingles  # noqa: E402
from thinking_corpus import REPO, TEACHER, THINK, rows  # noqa: E402


def tools_of(row):
    out = []
    for tool in row.get("tools") or []:
        function = dict(tool.get("function") or {})
        schema = function.pop("parameters_json", None)
        if schema and "parameters" not in function:
            function["parameters"] = json.loads(schema)
        out.append({"type": tool.get("type", "function"), "function": function})
    return out


def messages_of(row):
    """Template-ready messages, and (assistant turns, tool calls, reasoning chars)."""
    out, turns, calls, reasoning = [], 0, 0, 0
    for message in row.get("messages") or []:
        role, content = message.get("role"), message.get("content") or ""
        if not role:
            continue
        entry = {"role": role, "content": content}
        if role == "assistant":
            turns += 1
            match = THINK.match(content)
            if match:
                thought = edit_block(match.group(1))[0].strip()
                entry["content"] = content[match.end():]
                entry["reasoning_content"] = thought
                reasoning += len(thought)
            if message.get("tool_calls"):
                entry["tool_calls"] = []
                for call in message["tool_calls"]:
                    function = dict(call.get("function") or {})
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            function["arguments"] = json.loads(arguments)
                        except json.JSONDecodeError:
                            pass
                    entry["tool_calls"].append({"type": "function", "function": function,
                                                "id": call.get("id")})
                    calls += 1
        elif role == "tool":
            entry["name"] = message.get("name")
            entry["tool_call_id"] = message.get("tool_call_id")
        if role != "assistant" and not content.strip():
            continue
        out.append(entry)
    return out, turns, calls, reasoning


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configs", nargs="+",
                        default=["sft_balanced", "sft_agent", "sft_tools", "sft_glm_agent"])
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--tokens", type=int, default=6_000_000)
    parser.add_argument("--eval-every", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TEACHER)
    scored, related = build_banks()
    seen = set()
    stats = {c: dict(kept=0, tokens=0, no_tools=0, long=0, scored=0, related=0, duplicate=0,
                     calls=0, lengths=[]) for c in args.configs}
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
            tools = tools_of(row)
            messages, turns, calls, _ = messages_of(row)
            if not tools or not calls:
                s["no_tools"] += 1
                continue
            text = tok.apply_chat_template(messages, tools=tools, tokenize=False)
            length = len(tok(text, add_special_tokens=False)["input_ids"])
            if length > args.max_tokens:
                s["long"] += 1
                continue
            grams = shingles(text)
            if grams & scored:
                s["scored"] += 1
                continue
            if grams & related:
                s["related"] += 1
                continue
            key = fingerprint(text)
            if key in seen:
                s["duplicate"] += 1
                continue
            seen.add(key)
            split = "eval" if kept % args.eval_every == 0 else "train"
            ident = str(row.get("id") or row.get("parent_id") or kept)
            out.write(json.dumps({"doc_id": "agent:%s:%s" % (config, ident), "text": text,
                                  "split": split, "source": row.get("source") or config,
                                  "domain": "agent_tool", "tokens": length, "tool_calls": calls,
                                  "upstream_id": ident}, ensure_ascii=False) + "\n")
            kept += 1
            total += length
            s["kept"] += 1
            s["tokens"] += length
            s["calls"] += calls
            s["lengths"].append(length)
    for config, s in stats.items():
        lengths = sorted(s.pop("lengths"))
        s["median_tokens"] = lengths[len(lengths) // 2] if lengths else 0
        s["within_1024"] = sum(x <= 1024 for x in lengths)
        print("%-14s %s" % (config, json.dumps(s)))
    print("wrote %d documents, %d tokens -> %s" % (kept, total, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
