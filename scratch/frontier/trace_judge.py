"""A frontier judge over the teacher's own thinking traces (teacher_generate.py).

Grading already checked each trace's final answer against the reference; the student
copies the whole trace, so this asks what grading cannot: is the reasoning valid (not
right by luck or by an unjustified leap), is it efficient (no wandering, redundant
re-checking or repeated attempts), and would we want a small model to imitate it exactly.
Only graded-correct, finished, loop-free traces are sent, several to a request (the free
tier allows 1,000 requests a day). `collect` writes a ranked index of every trace --
grading and judge verdicts side by side, the traces themselves untouched -- and the kept ids.

    python scratch/frontier/trace_judge.py build --traces ../capture-data/teacher-gen-math.jsonl \\
        --output ../capture-data/frontier/trace-judge-requests.jsonl [--done <earlier responses>]
    python scratch/frontier/trace_judge.py collect --responses ../capture-data/frontier/trace-judge-responses.jsonl \\
        --traces ../capture-data/teacher-gen-math.jsonl --output ../capture-data/teacher-gen-keep.json
    (also writes teacher-gen-index.jsonl beside --output, best first)
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

INSTRUCTIONS = """You audit training data for a small (2B) language model that will learn to think by imitating \
these traces token for token. Each trace below (in <trace id=...> tags) is a stronger model's response to a math \
problem: its reasoning inside <think>...</think>, then its final answer. The final answer has already been checked \
against the reference and is correct; the reference is given. Judge each trace independently on the reasoning \
itself:

- "valid": "yes" if every step that the answer depends on is justified; "no" if it reaches the right answer by \
luck, guessing, an unjustified leap, a wrong intermediate result that happens to cancel, or by pattern-matching \
the reference; "unsure" if you cannot tell.
- "efficiency": 1-5. 5 = goes straight to a sound method and finishes; 3 = some detours or re-checking that \
earn their keep; 1 = wanders, tries many approaches aimlessly, re-derives the same thing, or pads.
- "clarity": 1-5, how easy the reasoning and the final explanation are to follow.
- "imitate": "yes" if you would want a small model to learn to think exactly like this trace, else "no".
- "issues": short strings, e.g. "lucky guess", "unjustified step", "arithmetic slip corrected late", \
"excessive re-checking", "aimless exploration", "repetition", "answer not explained", "hedging".

Return only JSON: {"verdicts": [one per trace: {"id": the trace id, "valid": "yes" | "no" | "unsure", \
"efficiency": 1-5, "clarity": 1-5, "imitate": "yes" | "no", "issues": [...]}]}"""
# The teacher's non-thinking answers (teacher_generate.py --nothink): the reasoning is the answer.
NOTHINK_INSTRUCTIONS = (INSTRUCTIONS
                        .replace("learn to think by imitating these traces", "learn to answer by imitating these responses")
                        .replace("its reasoning inside <think>...</think>, then its final answer",
                                 "answered directly, with no separate thinking section: its worked solution, then "
                                 "its final answer")
                        .replace("think exactly like this trace", "answer exactly like this response"))
assert NOTHINK_INSTRUCTIONS.count("directly") == 1 and "imitating these responses" in NOTHINK_INSTRUCTIONS


def trace_id(row):
    return "%s#%d" % (row["id"], row["seed"])


def verdicts_of(row):
    if "error" in row:
        return []
    try:
        reply = json.loads(re.sub(r"^```(json)?|```$", "", (row.get("content") or "").strip()))
    except ValueError:
        return []
    items = reply.get("verdicts") if isinstance(reply, dict) else None
    return [(str(v["id"]), v) for v in items or [] if isinstance(v, dict) and v.get("id")]


def read_traces(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except ValueError:  # the generator may be mid-write on the last line
            pass
    return rows


def score(v):
    """Rank key: valid and imitable first, then efficiency and clarity."""
    if v is None:
        return None
    return (2 * (v.get("valid") == "yes") + 2 * (v.get("imitate") == "yes")
            + int(v.get("efficiency") or 0) + int(v.get("clarity") or 0))


def build(args):
    done = set()
    for path in args.done or []:
        for row in map(json.loads, open(path, encoding="utf-8")):
            done |= {i for i, _ in verdicts_of(row)}
    traces = [r for r in read_traces(args.traces) if eligible(r) and trace_id(r) not in done]
    groups, group, size = [], [], 0
    for r in traces:  # several traces a request, up to a character budget
        if group and (len(group) == args.per_request or size + len(r["text"]) > args.max_chars):
            groups.append(group)
            group, size = [], 0
        group.append(r)
        size += len(r["text"])
    if group:
        groups.append(group)
    with open(args.output, "w", encoding="utf-8") as out:
        for g in groups:
            body = "\n\n".join('<trace id="%s">\nProblem: %s\nReference answer: %s\n\n%s\n</trace>'
                               % (trace_id(r), r["problem"], r["reference"], r["text"]) for r in g)
            out.write(json.dumps({"id": "trace-judge:" + trace_id(g[0]), "messages": [
                {"role": "system", "content": NOTHINK_INSTRUCTIONS if args.nothink else INSTRUCTIONS},
                {"role": "user", "content": body}],
                "max_tokens": 2000 + 600 * len(g), "reasoning": {"effort": "medium"},
                "response_format": {"type": "json_object"}}) + "\n")
    print("%d traces (%d already judged) in %d requests -> %s" % (len(traces), len(done), len(groups), args.output))


def eligible(r):
    """Graded correct, finished with its thought closed, and loop-free: what is judged."""
    return r["correct"] and r["finished"] and r["thought_closed"] and r["max_line_repeats"] < 5


def collect(args):
    seen, keep, counts, issues, verdict = set(), [], Counter(), Counter(), {}
    known = {trace_id(r) for r in read_traces(args.traces) if eligible(r)}
    for path in args.responses:
        for row in map(json.loads, open(path, encoding="utf-8")):
            for i, v in verdicts_of(row):
                if i not in known:  # a judge's id for no trace it was sent
                    counts["unknown id"] += 1
                    continue
                if i in seen:
                    continue
                seen.add(i)
                verdict[i] = v
                good = (v.get("valid") == "yes" and v.get("imitate") == "yes"
                        and int(v.get("efficiency") or 0) >= args.min_efficiency)
                counts["valid " + str(v.get("valid"))] += 1
                counts["imitate " + str(v.get("imitate"))] += 1
                counts["efficiency %s" % v.get("efficiency")] += 1
                issues.update(map(str, v.get("issues") or []))
                if good:
                    keep.append(i)
    args.output.write_text(json.dumps(sorted(keep), indent=0), encoding="utf-8")
    # Every trace, judged or not, best first: graded-wrong, unfinished and looping traces
    # stay in the index (unranked, at the end) with the reason they were not sent.
    index = []
    for r in read_traces(args.traces):
        i, v = trace_id(r), verdict.get(trace_id(r))
        index.append({"id": i, "line": len(index), "correct": r["correct"], "finished": r["finished"],
                      "max_line_repeats": r["max_line_repeats"], "tokens": r["tokens"],
                      "judged": v is not None, "kept": i in set(keep), "score": score(v),
                      **({k: v.get(k) for k in ("valid", "efficiency", "clarity", "imitate", "issues")} if v else {})})
    index.sort(key=lambda e: (e["score"] is None, -(e["score"] or 0), e["tokens"] or 0))
    for rank, e in enumerate(index, 1):
        e["rank"] = rank
    path = args.index or args.output.with_name("teacher-gen-index.jsonl")
    path.write_text("".join(json.dumps(e) + "\n" for e in index), encoding="utf-8")
    if args.exclusions:
        # The capture's ids (capture_math_gen: "tgen:" + trace id) of every trace not kept --
        # rejected or not yet judged (`--only-rejected`: judged and rejected) -- for
        # --exclude-documents, like the other lists.
        args.exclusions.write_text(json.dumps(sorted(args.prefix + e["id"] for e in index if not e["kept"]
                                                     and (e["judged"] or not args.only_rejected)),
                                              indent=0), encoding="utf-8")
    print("%d judged, %d kept -> %s\n%s\ncommon issues: %s"
          % (len(seen), len(keep), args.output, dict(sorted(counts.items())), issues.most_common(12)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--traces", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--done", type=Path, nargs="*", default=None)
    b.add_argument("--per-request", type=int, default=8)
    b.add_argument("--max-chars", type=int, default=60000)
    b.add_argument("--nothink", action="store_true", help="non-thinking answers: the judge's wording for them")
    c = sub.add_parser("collect")
    c.add_argument("--responses", type=Path, nargs="+", required=True)
    c.add_argument("--traces", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--min-efficiency", type=int, default=3)
    c.add_argument("--exclusions", type=Path, default=None, help="also write the not-kept capture ids here")
    c.add_argument("--prefix", default="tgen:", help="the capture's id prefix (tnothink: for non-thinking answers)")
    c.add_argument("--only-rejected", action="store_true", help="exclusions: judged and rejected only")
    c.add_argument("--index", type=Path, default=None, help="ranked index path (default teacher-gen-index.jsonl)")
    args = parser.parse_args()
    (build if args.command == "build" else collect)(args)


if __name__ == "__main__":
    main()
