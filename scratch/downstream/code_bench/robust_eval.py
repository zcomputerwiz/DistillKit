"""EvalPlus scoring that survives a sample killing its worker. Runs inside the sandbox.

`evalplus.evaluate` scores every sample in one ProcessPoolExecutor. When a completion's
test blows up a worker -- MBPP+'s extended inputs make a correct `combinations_with_
replacement` answer allocate without bound, and the kernel kills the process -- the pool
is broken and `evaluate()` waits on the remaining futures forever. The parent also holds
every task's expected outputs, about 1.5 GiB for MBPP+, which every forked worker shares.

Here each sample runs EvalPlus's own `check_correctness` in a process of its own, holding
only its task's expected outputs, under a wall-clock limit. A crash or timeout fails that
sample and nothing else. The scoring itself is EvalPlus's, unchanged.

    python robust_eval.py shard                     (image build: one file per task)
    python robust_eval.py run mbpp samples.jsonl out.json [workers]
"""
import json
import os
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

SHARDS = os.path.expanduser("~/shards")


def shard():
    from evalplus.data import (get_human_eval_plus, get_human_eval_plus_hash,
                               get_mbpp_plus, get_mbpp_plus_hash)
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.evaluate import get_groundtruth

    for dataset, problems, digest, special in (
            ("humaneval", get_human_eval_plus(), get_human_eval_plus_hash(), []),
            ("mbpp", get_mbpp_plus(), get_mbpp_plus_hash(), MBPP_OUTPUT_NOT_NONE_TASKS)):
        expected = get_groundtruth(problems, digest, special)
        os.makedirs(os.path.join(SHARDS, dataset), exist_ok=True)
        for task_id, problem in problems.items():
            name = task_id.replace("/", "_")
            with open(os.path.join(SHARDS, dataset, name + ".pkl"), "wb") as f:
                pickle.dump((problem, expected[task_id]), f)
        print(dataset, len(problems), "tasks sharded")


def one(dataset, task_id, solution_path):
    from evalplus.config import DEFAULT_GT_TIME_LIMIT_FACTOR, DEFAULT_MIN_TIME_LIMIT
    from evalplus.evaluate import check_correctness

    with open(os.path.join(SHARDS, dataset, task_id.replace("/", "_") + ".pkl"), "rb") as f:
        problem, expected = pickle.load(f)
    with open(solution_path, encoding="utf-8") as f:
        solution = f.read()
    result = check_correctness(dataset, 0, problem, solution, expected, base_only=False,
                               fast_check=True, identifier=task_id,
                               min_time_limit=DEFAULT_MIN_TIME_LIMIT,
                               gt_time_limit_factor=DEFAULT_GT_TIME_LIMIT_FACTOR)
    print(json.dumps({"base": result["base"][0], "plus": result["plus"][0]}))


def run(dataset, samples_path, out_path, workers):
    samples = [json.loads(line) for line in open(samples_path, encoding="utf-8")]
    os.makedirs("/tmp/solutions", exist_ok=True)

    def score(index_sample):
        index, sample = index_sample
        path = "/tmp/solutions/%d.py" % index
        with open(path, "w", encoding="utf-8") as f:
            f.write(sample["solution"])
        try:
            done = subprocess.run([sys.executable, __file__, "one", dataset, sample["task_id"], path],
                                  capture_output=True, text=True, timeout=300)
            line = done.stdout.strip().splitlines()[-1] if done.stdout.strip() else ""
            status = json.loads(line) if line.startswith("{") else {
                "base": "crash", "plus": "crash", "returncode": done.returncode}
        except subprocess.TimeoutExpired:
            status = {"base": "timeout", "plus": "timeout"}
        return dict(task_id=sample["task_id"], **status)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = []
        for number, row in enumerate(pool.map(score, enumerate(samples)), 1):
            results.append(row)
            if number % 25 == 0:
                print("%d/%d" % (number, len(samples)), flush=True)
    base = sum(r["base"] == "pass" for r in results)
    plus = sum(r["base"] == "pass" and r["plus"] == "pass" for r in results)
    summary = {"dataset": dataset, "n": len(results), "base_pass": base, "plus_pass": plus,
               "base_pass@1": base / len(results), "plus_pass@1": plus / len(results),
               "crashes": sum(r["base"] in ("crash", "timeout") for r in results)}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    if sys.argv[1] == "shard":
        shard()
    elif sys.argv[1] == "one":
        one(*sys.argv[2:5])
    else:
        run(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 3)
