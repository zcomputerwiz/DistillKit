# Assisted-by: Codex
"""Freeze and audit existing replay/pair inputs for discarded update diagnostics."""
import argparse
import ast
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
OUT = HERE / "influence-audit-20261008"
RUN = HERE / "completion-v2/completion/train.json"
EXCLUDE = HERE / "completion-gauntlet/dataset-audit/exclude-replay-next-20261008.json"
CODE = {"frontier-code-raw", "teacher-code", "r8-code-short-w8"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=True) + "\n", encoding="ascii")


def source_of(teacher, doc):
    return Path(teacher.cache._owner[doc][0]).name.removeprefix("teacher-cache-")


def teacher_for_run():
    from agentic_arm import teacher_for
    from teacher_kl import hedge_token_ids
    from transformers import AutoTokenizer
    args = read(RUN)["run_args"]
    assert args["strip_effort_nonthinking"] and not args["strip_effort_prompt"]
    assert args["min_answer_tokens"] == 2 and args["answer_weight"] == 8
    options = dict(paths=list(map(Path, args["teacher_cache"])), ce=args["ce_only_caches"],
                   assistant=args["assistant_only_caches"], kl=args["kl_only_caches"],
                   ul=args["unlikelihood_caches"], repeat=dict(s.rsplit("=", 1) for s in args["repeat"]))
    teacher = teacher_for(options)
    tokenizer = AutoTokenizer.from_pretrained(HERE / "merges-long1/u50", local_files_only=True)
    if args["suppress_hedges"]:
        teacher.suppress = hedge_token_ids(tokenizer)
    teacher.ids = [d for d in teacher.ids if d not in set(read(EXCLUDE))]
    return teacher, tokenizer, args


def target_stats(record, teacher, tokenizer, examples):
    from atlas import roles_of, ROLES
    doc = record["doc_ids"][0]
    source = source_of(teacher, doc)
    ids = record["input_ids"][0].numpy()
    w = record.get("weight", torch.ones_like(record["input_ids"], dtype=torch.float32))[0, :-1].numpy()
    labels = roles_of(ids)[1:]
    picks = record["topk_ids"][0, :-1].numpy()
    values = record["topk_logprobs"][0, :-1].numpy()
    active = w > 0
    if not np.isfinite(values[active]).all() or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("nonfinite targets or invalid weights: " + doc)
    if (picks[active] < 0).any() or (picks[active] >= 248320).any():
        raise ValueError("teacher token outside student vocabulary")
    if np.any(np.diff(np.sort(picks[active], axis=1), axis=1) == 0):
        raise ValueError("duplicate teacher top-k ID: " + doc)
    probs = np.exp(values)
    captured = probs.sum(1)
    top = picks[np.arange(len(w)), probs.argmax(1)]
    true_mass = np.where(picks == ids[1:, None], probs, 0).sum(1)
    neg = record.get("negative")
    neg = np.zeros(len(w), bool) if neg is None else neg[0, :-1].numpy()
    ce = 1. if record["ce_only"] else 0. if record["kl_only"] else .5
    kl = 0. if record["ce_only"] else 1. if record["kl_only"] else .5
    stats = {}
    for role, label in enumerate(ROLES):
        own = labels == role
        kw = w * own * ~neg * kl
        stats[label] = dict(targets=float((w * own).sum()), ce_weight=float((w * own * ce).sum()),
                            kl_weight=float(kw.sum()), teacher_top1_correct_weight=float((kw * (top == ids[1:])).sum()),
                            teacher_actual_absent_weight=float((kw * (true_mass == 0)).sum()),
                            premature_stop_weight=float((kw * np.isin(top, [248044, 248046]) *
                                                         ~np.isin(ids[1:], [248044, 248046])).sum()))
    context = sum(stats[r]["targets"] for r in ("system", "user", "tool-result"))
    if doc in teacher.assistant_only_ids and context:
        raise ValueError("scored conversational context: " + doc)
    if examples is not None:
        take = np.flatnonzero(active & ~neg & (top != ids[1:]) & (probs.max(1) > .9))[:2]
        for p in take:
            examples.append(dict(source=source, doc_id=doc, position=int(p),
                role=ROLES[int(labels[p])], prefix=tokenizer.decode(ids[max(0, p-40):p+1]),
                actual=tokenizer.decode([int(ids[p+1])]), teacher=tokenizer.decode([int(top[p])]),
                teacher_confidence=float(probs[p].max()), actual_captured_mass=float(true_mass[p])))
    return dict(doc_id=doc, source=source, width=len(ids), targets=float(w.sum()),
                ce_only=record["ce_only"], kl_only=record["kl_only"], roles=stats,
                min_captured_mass=float(captured[active].min()), max_captured_mass=float(captured[active].max()),
                input_sha256=hashlib.sha256(ids.tobytes()).hexdigest())


def pairs_audit(tokenizer):
    from completion_prepare import prefix_world, certify
    from completion_curriculum import render
    original = {r["pair_id"]: r for r in map(json.loads, (HERE / "completion-v2/data/pairs.jsonl").read_text().splitlines())}
    final = {r["pair_id"]: r for r in map(json.loads, (HERE / "completion-v2/data/pairs-final.jsonl").read_text().splitlines())}
    reference = list(map(json.loads, Path(read(RUN)["run_args"]["pairs"]).read_text().splitlines()))
    assert set(original) == set(final) == {r["pair_id"] for r in reference}
    result = []
    for row in reference:
        old = original[row["pair_id"]]
        current = final[row["pair_id"]]
        world, good = prefix_world(old)
        prompt, chosen = render(tokenizer, world, good)
        assert prompt == old["prompt"] == current["prompt"] and chosen == old["chosen"] == current["chosen"]
        reason, proof = certify(world, good, current["chosen"])
        assert reason is None and not proof.get("errors", []), row["pair_id"]
        for side in ("chosen", "rejected"):
            ids = tokenizer.encode(prompt + current[side], add_special_tokens=False)
            assert ids == row[side + "_ids"] and 0 < row[side + "_start"] < len(ids)
            assert math.isfinite(row["ref_" + side])
        start = row["chosen_start"]
        assert start == row["rejected_start"] == len(tokenizer.encode(prompt, add_special_tokens=False))
        assert row["chosen_ids"][:start] == row["rejected_ids"][:start]
        rejected_reason, rejected_proof = certify(world, good, current["rejected"])
        result.append(dict(pair_id=row["pair_id"], family=row["pair_id"].split(":")[2],
                           chosen_tokens=len(row["chosen_ids"])-start,
                           rejected_tokens=len(row["rejected_ids"])-start,
                           label=current["rejected_kind"], recertified_reason=rejected_reason,
                           chosen_proof=proof, rejected_proof=rejected_proof,
                           caveat="A valid unnecessary action is a cost/necessity contrast, not an invalid tool call."
                           if rejected_reason is None else None))
    return result


def prepare():
    if (OUT / "plan.json").exists():
        raise ValueError("frozen plan already exists; inspect it instead of rebuilding")
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    teacher, tokenizer, args = teacher_for_run()
    groups = read(args["ordered_batches"])["groups"]
    exclude = set(read(EXCLUDE))
    totals, visit = Counter(), Counter()
    records, examples = [], []
    candidates = defaultdict(list)
    for group in groups:
        for doc in group["documents"]:
            if doc in exclude:
                continue
            teacher.real_width[doc] = min(teacher.cap(doc), group["width"])
            batch = teacher.read_batch([doc], group["width"])
            summary = target_stats(batch, teacher, tokenizer, examples)
            records.append(summary)
            totals[summary["source"]] += summary["targets"]
            visit[doc] += 1
            if visit[doc] == 1:
                candidates[summary["source"]].append(doc)
        if len(records) % 10 == 0:
            print("Target audit documents", len(records), flush=True)
    selected = []
    for source, docs in sorted(candidates.items()):
        count = 3 if source in {"frontier-code-raw", "teacher-code"} else 2 if source == "r8-code-short-w8" else 1
        # Deterministic sample from retained observed exposure; no evaluation outcomes used.
        docs = sorted(docs, key=lambda d: hashlib.sha256(("influence-17/" + d).encode()).hexdigest())
        for doc in docs:
            cap = 8192 if source == "frontier-code-raw" else 4096
            width = min(teacher.cap(doc), cap)
            if teacher.doc_weight(doc, width) < 32:
                # Long agent prompts need their actual full prefix to reach assistant targets.
                width = min(teacher.cap(doc), 16384)
            if teacher.doc_weight(doc, width) < 32:
                continue
            teacher.real_width[doc] = width
            padded = -(-width // 128) * 128
            summary = target_stats(teacher.read_batch([doc], padded), teacher, tokenizer, None)
            selected.append(summary)
            if sum(r["source"] == source for r in selected) >= count:
                break
        if not any(r["source"] == source for r in selected):
            raise ValueError("no scored bounded diagnostic document for " + source)
    denominators = Counter()
    for r in selected:
        denominators[r["source"]] += r["targets"]
    for r in selected:
        r["coefficient"] = totals[r["source"]] / sum(totals.values()) * r["targets"] / denominators[r["source"]]
        r["preservation"] = r["source"] in CODE
    audited = pairs_audit(tokenizer)
    schedule = read(HERE / "completion-v2/data/pair-schedule.json")["indices"]
    all_rows = list(map(json.loads, Path(args["pairs"]).read_text().splitlines()))
    pair_ids = []
    for kind in ("aggregate", "aggregate_overlap", "recover_checked", "timeout_pending", "denied", "empty", "no_tool"):
        matches = [all_rows[i]["pair_id"] for i in schedule if all_rows[i]["pair_id"].split(":")[2] == kind]
        pair_ids.append(matches[len(matches)//2])
    hashes = {str(p): digest(p) for p in [RUN, EXCLUDE, Path(args["ordered_batches"]), Path(args["pairs"]),
        HERE / "completion-v2/data/pairs-final.jsonl", Path(__file__)]}
    for p in args["teacher_cache"]:
        hashes[str(Path(p) / "manifest.json")] = digest(Path(p) / "manifest.json")
    plan = dict(version=1, purpose="discarded diagnostics; no training or promotion",
                checkpoint=str(HERE / "completion-v2/completion/checkpoints/smoke-r1-1-gr-s25-csa2"),
                state=str(HERE / "completion-v2/completion/checkpoints/state-step-00000040"),
                replay=selected, pair_ids=pair_ids, source_targets=dict(totals),
                weighting="Retain source shares from historical prefix after confirmed exclusions; within sampled source, token-weighted.",
                limits=["Small deterministic training-data sample; not an independent outcome evaluation.",
                        "Raw code capped at 8K, other sources at 4K or 16K where assistant targets occur later.",
                        "Selected preference pairs are stratified, not a reconstruction of their historical frequency.",
                        "No root loss, balancing rule or gradient surgery applied."], input_sha256=hashes)
    write(OUT / "target-audit.json", dict(documents=records, selected=selected, source_targets=dict(totals),
        confident_disagreements=examples[:80], total_targets=sum(totals.values()),
        excluded_historical_visits=sum(d in exclude for g in groups for d in g["documents"]),
        caveat="Teacher disagreement with a valid sampled token does not by itself establish a bad target."))
    write(OUT / "pair-execution-audit.json", dict(rows=audited, count=len(audited),
        valid_uncertified_contrasts=[r["pair_id"] for r in audited if r["recertified_reason"] is None]))
    write(OUT / "plan.json", plan)
    teacher.close()
    print("Frozen", len(selected), "replay documents and", len(pair_ids), "pairs; all", len(audited), "pair prefixes/targets checked", flush=True)


def code_execution():
    from code_dataset_audit import DATA
    sys.path.insert(0, str(ROOT / "scratch/downstream/code_bench"))
    from verify_code import sandbox, solution_of
    excluded = set(read(EXCLUDE))
    selected = {r["doc_id"] for r in read(OUT / "plan.json")["replay"]}
    samples, details = [], []
    for source, filename, prompts_file in (("teacher-code", "teacher-code-verified.jsonl", "code-prompts-teacher.jsonl"),
        ("r8-code-short-w8", "onpolicy-r8-code-short.jsonl", "code-prompts-r8.jsonl")):
        prompts = {r["doc_id"]: r for r in map(json.loads, (DATA / prompts_file).read_text().splitlines())}
        pool = list(map(json.loads, (DATA / filename).read_text().splitlines()))
        retained = {r["doc_id"] for r in read(ROOT.parent / ("teacher-cache-"+source) / "manifest.json")["documents"]
                    if r["split"] == "train" and r["doc_id"] not in excluded}
        pool = [r for r in pool if ("tcode:"+r["doc_id"] if source == "teacher-code" else r["doc_id"]) in retained]
        ordered = sorted(pool, key=lambda r: (not (("tcode:"+r["doc_id"] if source == "teacher-code" else r["doc_id"]) in selected),
                           hashlib.sha256(("execution-19/"+r["doc_id"]).encode()).hexdigest()))[:8]
        for r in ordered:
            doc = "tcode:"+r["doc_id"] if source == "teacher-code" else r["doc_id"]
            key = re.sub(r"^onpolicy:|:s\d+$|:greedy$", "", r["doc_id"])
            reply = r["text"][r["prompt_chars"]:]
            final = reply.split("</think>")[-1]
            explicit = re.findall(r"```(?:python|py)\s*\n(.*?)```", final, re.S)
            solution = explicit[0] if explicit else solution_of(reply)
            tree = ast.parse(solution)
            variants = [("served", solution)]
            # Deliberately destructive return mutation checks that the tests exercise outputs.
            altered = copy.deepcopy(tree)
            returns = [n for n in ast.walk(altered) if isinstance(n, ast.Return) and n.value is not None]
            if returns:
                for n in returns:
                    n.value = ast.Constant(None)
                variants.append(("null_returns", ast.unparse(ast.fix_missing_locations(altered))))
            altered = copy.deepcopy(tree)
            changes = {ast.Lt: ast.LtE, ast.Gt: ast.GtE, ast.LtE: ast.Lt, ast.GtE: ast.Gt, ast.Eq: ast.NotEq}
            boundary = next((n for n in ast.walk(altered) if isinstance(n, ast.Compare) and type(n.ops[0]) in changes), None)
            if boundary is not None:
                boundary.ops[0] = changes[type(boundary.ops[0])]()
                variants.append(("comparison_boundary", ast.unparse(ast.fix_missing_locations(altered))))
            for name, code in variants:
                samples.append(dict(id=doc+"#"+name, solution=code, test=prompts[key]["tests"]))
            details.append(dict(doc_id=doc, source=source, selected_gradient_sample=doc in selected,
                                original_verified=r.get("verified"), variants=[v[0] for v in variants],
                                solution=solution, tests=prompts[key]["tests"], prompt=prompts[key].get("prompt"),
                                served_extraction_matches_historical_last=solution.strip() == (solution_of(reply) or "").strip()))
    print("Executing", len(samples), "code/controlled mutation checks in existing sandbox", flush=True)
    results = sandbox(samples, workers=3, seconds=20)
    assert set(results) == {r["id"] for r in samples}
    for r in details:
        r["statuses"] = {v: results[r["doc_id"]+"#"+v] for v in r["variants"]}
    write(OUT / "code-execution-audit.json", dict(rows=details, counts=dict(Counter(results.values())),
        caveats=["Surviving comparison mutations may be equivalent; manual review is required.",
                 "Existing tests are reexecuted; passing is not a completeness guarantee.",
                 "No source label, frozen result or cached teacher target changed."]))
    print("Code checks", dict(Counter(results.values())), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "code"))
    a = p.parse_args()
    (prepare if a.mode == "prepare" else code_execution)()
