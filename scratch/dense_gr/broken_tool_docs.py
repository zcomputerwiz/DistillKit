"""Every captured document whose upstream trace had tool calls the renderer dropped.

`prepare_corpus.render_messages` and `thinking_corpus.native` kept role and content only,
so any trace with `tool_calls` lost them. This finds those documents -- the 5M corpus's
agent traces and their de-hedged (:dh) and think-first (:tf) copies, and the thinking
corpus's rows whose upstream messages carried calls -- and writes one exclusion list,
merged with the contamination list, for every run that reads those captures.

    python scratch/dense_gr/broken_tool_docs.py --output ../capture-data/exclude-broken-tools.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from thinking_corpus import rows  # noqa: E402

ROOT = Path("D:/DeepThought/Projects/HybridModel/capture-data")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    broken = set()
    # The 5M corpus: its agent/tool sources, and every copy made of them.
    agent = {json.loads(l)["doc_id"] for l in open(ROOT / "run5m.jsonl", encoding="utf-8")
             if "agent" in (json.loads(l).get("source") or "") or "tool" in (json.loads(l).get("source") or "")}
    broken |= agent
    for name, key in (("dehedged-5m.jsonl", "rewritten_from"), ("think-first-5m.jsonl", "remixed_from")):
        for line in open(ROOT / name, encoding="utf-8"):
            row = json.loads(line)
            if row[key] in agent:
                broken.add(row["doc_id"])
    # The thinking corpus: rows whose upstream messages had tool calls.
    thinking = [json.loads(l) for l in open(ROOT / "thinking-code-math.jsonl", encoding="utf-8")]
    wanted = {r["doc_id"].split(":", 2)[2]: r["doc_id"] for r in thinking
              if r["doc_id"].startswith("think:sft_reasoning:")}
    with_calls = set()
    for row in rows("sft_reasoning"):
        ident = str(row.get("id") or row.get("parent_id") or "")
        if ident in wanted and any(m.get("tool_calls") for m in row.get("messages") or []):
            with_calls.add(wanted[ident])
    broken |= with_calls
    contamination = set(json.load(open(ROOT / "exclude-for-think-first.json", encoding="utf-8")))
    json.dump(sorted(broken | contamination), open(args.output, "w"), indent=1)
    print("5M agent documents %d, with copies %d; thinking-corpus rows with dropped calls %d; "
          "plus %d contamination -> %d ids in %s"
          % (len(agent), len(broken) - len(with_calls), len(with_calls), len(contamination),
             len(broken | contamination), args.output))


if __name__ == "__main__":
    main()
