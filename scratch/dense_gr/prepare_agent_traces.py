"""SmolDataEnvs agent traces as capture input: one document per rollout.

FineEnvs/SmolDataEnvs-multiharness-sft is Qwen3.8-27B -- this project's teacher -- solving
data-analysis and software tasks inside Claude Code, OpenCode, Codex and mini-swe-agent,
successful rollouts only, published one row per assistant turn with the history as its
prompt. Those rows repeat the same history again and again (Claude Code: median 16.8K
tokens of prompt for a median 240-token completion), so each rollout is taken once, at
its last turn, whose history holds every earlier turn: trained with
`--assistant-only-caches`, all of its assistant turns are scored in one sequence.

Rendered with the served chat template (the teacher's, which the student adopted), which
keeps every turn's reasoning across tool rounds. A tenth of the tasks, chosen by a hash
of the task id, are held out in every harness, so no task's other harnesses leak into
training.

    python scratch/dense_gr/prepare_agent_traces.py --output ../capture-data/agent-smol.jsonl
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

from long_context_probe import AGENT, TOKENIZER, messages_of, plain

HARNESSES = ("claude-code", "opencode", "codex", "mini-swe-agent")


def held_out(task_id, share):
    return int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--eval-share", type=float, default=0.1)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    import pandas as pd
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    counts = {}
    with open(args.output, "w", encoding="utf-8") as out:
        for harness in HARNESSES:
            frame = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(AGENT / harness / "*.parquet"))))
            frame = frame.assign(history=frame["prompt"].map(lambda p: len(plain(p))))
            for _, row in frame.sort_values("history").groupby("rollout_id").tail(1).iterrows():
                text = tokenizer.apply_chat_template(messages_of(row), tools=plain(row["tools"]), tokenize=False)
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                split = "eval" if held_out(row["task_id"], args.eval_share) else "train"
                out.write(json.dumps({"doc_id": "%s:%s" % (harness, row["rollout_id"]), "split": split,
                                      "input_ids": ids[:args.max_length], "harness": harness,
                                      "task_id": row["task_id"], "length": len(ids)}) + "\n")
                key = (harness, split)
                n, tokens, cut = counts.get(key, (0, 0, 0))
                counts[key] = (n + 1, tokens + min(len(ids), args.max_length), cut + (len(ids) > args.max_length))
    for (harness, split), (n, tokens, cut) in sorted(counts.items()):
        print("%-15s %-5s %5d rollouts %11d tokens, %d cut at %d" % (harness, split, n, tokens, cut, args.max_length))
    print("total %d tokens -> %s" % (sum(t for _, t, _ in counts.values()), args.output))


if __name__ == "__main__":
    main()
