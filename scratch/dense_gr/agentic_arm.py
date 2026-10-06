# Assisted-by: Codex
"""Plan and run a bounded agentic SFT arm with the existing replay recipe.

No model source changes. All work is local. Child failures stop the pipeline.
Run `plan` before `run`; the serialized exact argv is the run specification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))


def recipe(data, out, new_repeat=1, steps=50, control=False, code_multiplier=1, rate_scale=.55, balanced=False):
    names = ["agent-smol-a", "agent-smol-b", "frontier-qa2", "frontier-code-raw", "frontier-tools",
             "r8-code-short-w8", "curriculum-v4-w8", "thinking-w8", "think-first-w8", "expand-code-w8",
             "general-pilot-w8", "teacher-math-gen", "onpolicy-r6-loop", "loop-check",
             "teacher-nothink-math", "teacher-code"]
    paths = {n: str((ROOT.parent / ("teacher-cache-" + n)).resolve()) for n in names}
    factors = dict(zip(names, [3, 3, 1, 1, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 8]))
    for name in ("frontier-code-raw", "r8-code-short-w8", "expand-code-w8", "teacher-code"):
        factors[name] *= code_multiplier
    cache = str((data / "cache").resolve())
    all_caches = list(paths.values()) + ([] if control else [cache])
    assistant = [paths[n] for n in ("agent-smol-a", "agent-smol-b", "frontier-tools", "frontier-qa2")]
    ce = [paths["r8-code-short-w8"]]
    repeat = {paths[n]: factors[n] for n in names}
    if not control:
        assistant.append(cache)
        ce.append(cache)
        repeat[cache] = new_repeat
    argv = [str(HERE / "smoke_train.py"), "--init-from", str(HERE / "merges-long1/u50"),
            "--inherit", "--tensor-parallel", "--embedding-on", "away", "--checkpoint-layers",
            "--checkpoint-selection-cache", "--shared-head-loss", "--streaming-head-loss",
            "--teacher-cache", *all_caches,
            "--assistant-only-caches", *assistant, "--ce-only-caches", *ce,
            "--kl-only-caches", paths["frontier-code-raw"],
            "--unlikelihood-caches", paths["onpolicy-r6-loop"], paths["loop-check"],
            "--repeat", *[f"{p}={n}" for p, n in repeat.items()],
            "--pad-to-block", "--head-chunk", "512",
            "--answer-spans", str(ROOT.parent / "capture-data/frontier-qa2.jsonl"), "--answer-weight", "8",
            "--lr-scale", r"linear_attn\.(A_log|dt_bias|in_proj_a)=0.1", "--strip-effort-nonthinking",
            "--exclude-documents", str(ROOT.parent / "capture-data/exclude-long-r5.json"), "--suppress-hedges",
            "--teacher-weight", "0.5", "--teacher-max-length", "32768", "--kl-chunk", "64",
            "--min-answer-tokens", "2", "--micro-tokens", "32768", "--accumulate", "2",
            "--tokens", "10000000", "--max-steps", str(steps), "--warmup", "10",
            "--decay-fraction", "0.5", "--decay-floor", "0.05",
            "--lr-depth-ramp", str(rate_scale), str(rate_scale), "--evaluate-windows", "64",
            "--evaluate-every", str(steps), "--report-every", "5", "--save-every", str(steps),
            "--seed", "25", "--checkpoints", str(out / "checkpoints"), "--output", str(out / "train.json")]
    if balanced:
        argv += ["--balanced-prefix-batches", str(steps * 2)]
    return argv, dict(paths=all_caches, assistant=assistant, ce=ce, repeat=repeat,
                      kl=[paths["frontier-code-raw"]], ul=[paths["onpolicy-r6-loop"], paths["loop-check"]])


def teacher_for(options):
    from transformers import AutoTokenizer
    from teacher_kl import CachedTeacher
    from smoke_train import ANSWER_MARKER, EFFORT_PROMPT, excluded_documents
    tok = AutoTokenizer.from_pretrained(HERE / "merges-long1/u50")
    encode = lambda s: tok.encode(s, add_special_tokens=False)
    spans = {}
    for line in (ROOT.parent / "capture-data/frontier-qa2.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("answer_spans"):
            spans[row["doc_id"]] = row["answer_spans"]
    header, effort = encode("<|im_start|>system\n"), encode(EFFORT_PROMPT)
    whole = encode("<|im_start|>system\n%s<|im_end|>\n" % EFFORT_PROMPT)
    heading = header + effort + encode("\n\n")
    teacher = CachedTeacher(options["paths"], device="cpu", seed=25, max_length=32768,
        answer_marker=encode(ANSWER_MARKER), min_answer_tokens=2,
        exclude=excluded_documents(ROOT.parent / "capture-data/exclude-long-r5.json"),
        strip_prefix=[(whole, 0, len(whole)), (heading, len(header), len(heading))],
        strip_nonthinking=(encode(ANSWER_MARKER), encode("\n<think>\n\n</think>")),
        ce_only=options["ce"], assistant_only=options["assistant"],
        kl_only=options["kl"], unlikelihood=options["ul"],
        turn_close=tok.convert_tokens_to_ids("<|im_end|>"), think_close=tok.convert_tokens_to_ids("</think>"),
        repeat=options["repeat"], answer_spans=spans, answer_weight=8.)
    teacher.pad_blocks = True
    return teacher


def measure(teacher, steps, balanced=False):
    import numpy as np
    groups = teacher._groups(1, 128, 32768)
    totals, prefix = Counter(), Counter()
    for i, target in ((i, totals) for i in range(len(groups))):
        group, width = groups[i]
        for doc in group:
            source = Path(teacher.cache._owner[doc][0]).name
            target[source] += teacher.doc_weight(doc, min(teacher.real_width[doc], width))
    rng = np.random.default_rng(25)
    from training_state import coverage_order
    order = coverage_order(teacher, groups, rng, steps * 2) if balanced else []
    while len(order) < steps * 2:
        order.extend(rng.permutation(len(groups)).tolist())
    visits = Counter()
    for i in order[:steps * 2]:
        group, width = groups[i]
        for doc in group:
            source = Path(teacher.cache._owner[doc][0]).name
            prefix[source] += teacher.doc_weight(doc, min(teacher.real_width[doc], width))
            if doc.startswith(("agentic-v1:", "agentic-v2:")):
                visits[doc] += 1
    return dict(epoch_weighted_targets=dict(totals), prefix_weighted_targets=dict(prefix),
                prefix_total=sum(prefix.values()), prefix_new_fraction=prefix["cache"] / max(1, sum(prefix.values())),
                prefix_unique_new_documents=len(visits), prefix_max_new_visits=max(visits.values(), default=0),
                batches=len(groups), shapes=len({(len(g), w) for g, w in groups}))


def plan(args):
    if args.output.exists():
        raise ValueError("refuse to overwrite run directory")
    args.output.mkdir(parents=True)
    tuning = dict(code_multiplier=args.code_multiplier, rate_scale=args.rate_scale, balanced=args.balanced)
    _, options = recipe(args.data, args.output / "agentic", steps=args.steps, **tuning)
    print("Measuring replay and new-target mass", flush=True)
    teacher = teacher_for(options)
    initial = measure(teacher, args.steps, args.balanced)
    new = initial["epoch_weighted_targets"]["cache"]
    old = sum(initial["epoch_weighted_targets"].values()) - new
    factor = max(1, round(old * args.new_share / (1 - args.new_share) / new))
    # Masking and target counts do not depend on repetition. Reuse this CPU
    # cache scan; change only the document multiset used to build groups.
    original_ids = list(teacher.ids)
    new_ids = set(teacher.cache.caches[-1].document_ids("train"))
    plans = {}
    for arm in ("agentic", "replay"):
        print("Planning " + arm, flush=True)
        argv, options = recipe(args.data, args.output / arm, factor, args.steps, arm == "replay", **tuning)
        teacher.ids = [d for d in original_ids for _ in range(
            (factor if arm == "agentic" else 0) if d in new_ids else 1)]
        plans[arm] = dict(argv=argv, mixture=measure(teacher, args.steps, args.balanced))
        if args.balanced:
            observed = plans[arm]["mixture"]["prefix_weighted_targets"]
            if any(observed.get(Path(p).name, 0) <= 0 for p in options["paths"]):
                raise ValueError("balanced prefix failed to score every source")
    teacher.cache.close()
    result = dict(start_checkpoint=str(HERE / "merges-long1/u50"), steps=args.steps,
                  new_repeat=factor, data=str(args.data.resolve()), arms=plans, tuning=tuning,
                  requested_new_share=args.new_share,
                  limitation="Equal optimizer steps and recipe, not identical token counts or replay documents after regrouping.")
    (args.output / "plan.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v["mixture"] for k, v in plans.items()}, indent=2))


def run(args):
    os.chdir(ROOT)
    spec = json.loads((args.output / "plan.json").read_text())
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    status = args.output / "status.json"

    def invoke(label, argv):
        status.write_text(json.dumps(dict(stage=label, status="running", pid=os.getpid())), encoding="utf-8")
        with (args.output / (label + ".log")).open("w", encoding="utf-8") as log:
            child = subprocess.run([sys.executable, *map(str, argv)], env=env, cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT)
        if child.returncode:
            status.write_text(json.dumps(dict(stage=label, status="failed", exit_code=child.returncode)), encoding="utf-8")
            raise SystemExit(child.returncode)

    # Training first; no background probes compete for CPU or GPU.
    for arm in ("agentic", "replay"):
        (args.output / arm).mkdir(exist_ok=True)
        invoke("train-" + arm, spec["arms"][arm]["argv"])
    checkpoints = {"base": spec["start_checkpoint"], **{
        a: str(args.output / a / "checkpoints/smoke-r1-1-gr-s25-csa2") for a in ("agentic", "replay")}}
    for arm, checkpoint in checkpoints.items():
        invoke(f"live-{arm}", [HERE / "agentic_live_eval.py", "--checkpoint", checkpoint,
               "--data", Path(spec["data"]) / "trajectories.jsonl", "--output", args.output / f"live-{arm}.json"])
        if json.loads((Path(spec["data"]) / "summary.json").read_text()).get("version", 1) >= 2:
            invoke(f"live-{arm}-v1", [HERE / "agentic_live_eval.py", "--checkpoint", checkpoint,
                   "--data", HERE / "agentic-v1-verified/trajectories.jsonl",
                   "--output", args.output / f"live-{arm}-v1.json"])
        for suite, fixture in (("transfer", Path(spec["data"]) / "heldout-frozen.json"),
                               ("original", ROOT / "scratch/csa2-eval/tool-behavior/frozen.json"),
                               ("post-tool", ROOT / "scratch/csa2-eval/tool-behavior/post-tool-frozen.json")):
            invoke(f"eval-{arm}-{suite}", [HERE / "tool_behavior_eval.py", "run", "--checkpoint", checkpoint,
                "--arm", arm, "--fixture", fixture, "--device", "cuda:0", "--batch", "8", "--new", "512",
                "--output", args.output / f"eval-{arm}-{suite}"])
    ledger = [HERE / "atlas.py", "nll", "--domains", ROOT / "scratch/csa2-eval/atlas/domains-qa32768-20261005.pt",
              "--output-dir", args.output / "ledger", "--save-token-evidence"]
    for arm, checkpoint in checkpoints.items():
        ledger += ["--arm", f"{arm}={checkpoint}"]
    invoke("retention-ledger", ledger)
    invoke("retention-paired", [HERE / "atlas_compare.py", "--evidence", args.output / "ledger/token_evidence.npz",
            "--reference", "replay", "--arm", "agentic", "--output", args.output / "ledger/paired.json"])
    invoke("retention-proxy", [HERE / "merge_proxy.py", *[f"{a}={p}" for a, p in checkpoints.items()],
                               "--count", "256", "--output", args.output / "proxy.json"])
    status.write_text(json.dumps(dict(status="complete", promotion="not approved; review retention and tool results")), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["plan", "run"])
    p.add_argument("--data", type=Path, default=HERE / "agentic-v1-verified")
    p.add_argument("--output", type=Path, default=HERE / "agentic-pilot-v1")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--new-share", type=float, default=.15)
    p.add_argument("--code-multiplier", type=int, default=1)
    p.add_argument("--rate-scale", type=float, default=.55)
    p.add_argument("--balanced", action="store_true")
    a = p.parse_args()
    if not 0 < a.new_share < 1 or a.code_multiplier < 1 or not 0 < a.rate_scale <= 1 or a.steps < 1:
        p.error("invalid share, code multiplier, rate scale, or step count")
    a.output = a.output.resolve()
    {"plan": plan, "run": run}[a.command](a)
