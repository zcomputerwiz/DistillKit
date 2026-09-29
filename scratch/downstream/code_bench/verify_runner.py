"""Run pytest suites against model-written solutions. Runs inside the sandbox only.

Each sample is `{"id", "solution", "test"}`: the solution is written to solution.py and
the tests (which `from solution import ...`) to test_solution.py in a fresh directory,
and pytest runs in a process of its own under a wall-clock limit, so a hang or a crash
fails that sample and nothing else.

    python verify_runner.py samples.jsonl results.jsonl [workers] [seconds]
"""
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor


def run(sample, seconds):
    with tempfile.TemporaryDirectory() as work:
        with open(os.path.join(work, "solution.py"), "w", encoding="utf-8") as f:
            f.write(sample["solution"])
        with open(os.path.join(work, "test_solution.py"), "w", encoding="utf-8") as f:
            f.write(sample["test"])
        try:
            done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                                   "test_solution.py"], cwd=work, capture_output=True, text=True,
                                  timeout=seconds)
            status = "passed" if done.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            status = "timeout"
    return {"id": sample["id"], "status": status}


def main():
    source, target = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 20
    samples = [json.loads(line) for line in open(source, encoding="utf-8")]
    with ThreadPoolExecutor(workers) as pool, open(target, "w", encoding="utf-8") as out:
        for result in pool.map(lambda s: run(s, seconds), samples):
            out.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
