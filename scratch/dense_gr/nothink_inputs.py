"""Non-thinking examples: the teacher's own non-thinking answers, as capture input.

Two candidates were compared on the teacher's non-thinking view (nothink_pilot.py, 200
problems): its thinking-mode answers with the thought dropped, and its own non-thinking
answers. The dropped-thought answers carry leaps -- a median of 2 tokens an answer the
non-thinking teacher gives under 1% (worst token ~1e-8): steps it only resolved while
thinking, which would teach the student to state what it cannot derive. Its own non-thinking
answers show the work instead (2.4x longer) and were all correct on those problems. So these
are the teacher's non-thinking samples (teacher_generate.py --nothink), graded correct,
finished, loop-free; split as the problem's thinking traces (capture_inputs.split_of on the
trace id), so no problem crosses it.

    python scratch/dense_gr/nothink_inputs.py problems --count 2500       (from the kept traces)
    python scratch/dense_gr/teacher_generate.py --nothink --problems ../capture-data/teacher-nothink-problems.jsonl \\
        --output ../capture-data/teacher-gen-nothink.jsonl
    python scratch/dense_gr/nothink_inputs.py inputs --output ../capture-data/teacher-nothink-math.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "frontier"))
from capture_inputs import split_of  # noqa: E402
from nothink_pilot import C, TEACHER, trace_id  # noqa: E402


def problems(args):
    """A random `count` of the kept traces' problems, the pilot's first (their answers exist)."""
    keep = set(json.loads((C / "teacher-gen-keep.json").read_text(encoding="utf-8")))
    rows = [r for r in map(json.loads, open(C / "teacher-gen-math.jsonl", encoding="utf-8")) if trace_id(r) in keep]
    pilot = [json.loads(line)["id"] for line in open(C / "nothink-pilot-problems.jsonl", encoding="utf-8")]
    rest = [r for r in rows if r["id"] not in set(pilot)]
    picked = pilot + [r["id"] for r in random.Random(args.seed).sample(rest, args.count - len(pilot))]
    by_id = {r["id"]: r for r in rows}
    with open(C / "teacher-nothink-problems.jsonl", "w", encoding="utf-8") as out:
        for pid in picked:
            r = by_id[pid]
            out.write(json.dumps({"id": pid, "problem": r["problem"], "answer": r["reference"]}) + "\n")
    # The pilot's native answers are the first of them: teacher_generate resumes past these.
    target = C / "teacher-gen-nothink.jsonl"
    if not target.exists():
        target.write_text((C / "nothink-pilot-native.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    print("%d problems (%d from the pilot) -> %s" % (len(picked), len(pilot), C / "teacher-nothink-problems.jsonl"))


def inputs(args):
    from transformers import AutoTokenizer

    wanted = {json.loads(line)["id"] for line in open(args.problems, encoding="utf-8")}
    answers = [json.loads(line) for line in open(args.answers, encoding="utf-8")]
    missing = wanted - {row["id"] for row in answers}
    if missing:
        raise SystemExit("%d of %d problems have no answer, e.g. %s; run teacher_generate.py --nothink "
                         "again (it resumes)" % (len(missing), len(wanted), sorted(missing)[:3]))
    tok = AutoTokenizer.from_pretrained(TEACHER)
    end = tok.convert_tokens_to_ids("<|im_end|>")
    n = tokens = total = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for row in answers:
            total += 1
            if not (row["correct"] and row["finished"] and row["max_line_repeats"] < 5):
                continue
            tid = trace_id(row)
            ids = (tok(row["prompt"], add_special_tokens=False)["input_ids"]
                   + tok(row["text"].strip(), add_special_tokens=False)["input_ids"] + [end])
            out.write(json.dumps({"doc_id": "tnothink:" + tid, "split": split_of(tid), "input_ids": ids}) + "\n")
            n += 1
            tokens += len(ids)
    print("%d of %d non-thinking answers kept, %d tokens -> %s" % (n, total, tokens, args.output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("problems")
    p.add_argument("--count", type=int, default=2500)
    p.add_argument("--seed", type=int, default=6)
    i = sub.add_parser("inputs")
    i.add_argument("--answers", type=Path, default=C / "teacher-gen-nothink.jsonl")
    i.add_argument("--problems", type=Path, default=C / "teacher-nothink-problems.jsonl")
    i.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (problems if args.command == "problems" else inputs)(args)


if __name__ == "__main__":
    main()
