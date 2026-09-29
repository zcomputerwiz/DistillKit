"""Verify code rollouts against their pytest suites, in the sandbox, and mark each one.

Rollouts are the student's answers to code prompts (`code_prompts.py`); the tests come
from the prompts file by doc id. The final ```python block of the reply (after any
thought) is the solution. Samples go into a `code-verify-sandbox` container with no
network and no host mounts -- `docker cp` in, results out -- where `verify_runner.py`
runs each under its own time limit. Output: the rollouts with `verified` set to
passed / failed / timeout / no_code.

    python scratch/downstream/code_bench/verify_code.py --prompts <prompts.jsonl> \\
        --rollouts <rollouts.jsonl> --output <verified.jsonl>
"""
import argparse
import json
import re
import subprocess
import tempfile
import uuid
from pathlib import Path

BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def solution_of(text):
    """The last fenced block of the reply, after any thought; None without one."""
    reply = text.split("</think>")[-1]
    blocks = BLOCK.findall(reply)
    return blocks[-1] if blocks else None


def sandbox(samples, workers=3, seconds=20):
    """Run samples in a fresh container; returns {id: status}."""
    name = "code-verify-" + uuid.uuid4().hex[:8]
    with tempfile.TemporaryDirectory() as work:
        source = Path(work) / "samples.jsonl"
        with open(source, "w", encoding="utf-8") as out:
            for sample in samples:
                out.write(json.dumps(sample) + "\n")
        subprocess.run(["docker", "create", "--name", name, "--network", "none", "--memory", "3700m",
                        "--pids-limit", "512", "--cpus", "8", "code-verify-sandbox", "timeout", "14400",
                        "python", "verify_runner.py", "/home/runner/samples.jsonl",
                        "/home/runner/results.jsonl", str(workers), str(seconds)],
                       check=True, capture_output=True)
        try:
            subprocess.run(["docker", "cp", str(source), name + ":/home/runner/samples.jsonl"], check=True)
            subprocess.run(["docker", "start", "-a", name], check=True)
            target = Path(work) / "results.jsonl"
            subprocess.run(["docker", "cp", name + ":/home/runner/results.jsonl", str(target)], check=True)
            return {r["id"]: r["status"] for r in map(json.loads, open(target, encoding="utf-8"))}
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    tests = {}
    for line in open(args.prompts, encoding="utf-8"):
        row = json.loads(line)
        if row.get("tests"):
            tests[row["doc_id"]] = row["tests"]
    rows, samples = [], []
    for line in open(args.rollouts, encoding="utf-8"):
        row = json.loads(line)
        rows.append(row)
        key = re.sub(r"^onpolicy:|:s\d+$|:greedy$", "", row["doc_id"])
        code = solution_of(row["text"][row["prompt_chars"]:])
        if key not in tests:
            row["verified"] = None
        elif code is None or not row["finished"]:
            row["verified"] = "no_code" if row["finished"] else "cut"
        else:
            samples.append({"id": row["doc_id"], "solution": code, "test": tests[key]})
    status = sandbox(samples) if samples else {}
    counts = {}
    with open(args.output, "w", encoding="utf-8") as out:
        for row in rows:
            if row["doc_id"] in status:
                row["verified"] = status[row["doc_id"]]
            counts[row["verified"]] = counts.get(row["verified"], 0) + 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("verified %d rollouts: %s -> %s" % (len(rows), json.dumps(counts), args.output))


if __name__ == "__main__":
    main()
