# Assisted-by: Codex
"""Audit actual sampled objectives and the existing proxy banks. CPU only."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import numpy as np
import torch
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
from atlas import roles_of, ROLES
from transformers import AutoTokenizer


def proxies(out):
    from merge_proxy import problems
    from teacher_kl import CachedTeacher
    from smoke_train import ANSWER_MARKER, EFFORT_PROMPT
    from distillkit.offline_cache import OfflineTeacherCache
    tok = AutoTokenizer.from_pretrained(HERE / "merges-long1/u50")
    encode = lambda s: tok.encode(s, add_special_tokens=False)
    header, effort = encode("<|im_start|>system\n"), encode(EFFORT_PROMPT)
    whole = encode("<|im_start|>system\n%s<|im_end|>\n" % EFFORT_PROMPT)
    heading = header + effort + encode("\n\n")
    domains, role_sets, report = {}, {}, {}
    for name, source in (("code", "expand-code"), ("thinking", "thinking")):
        cache = HERE.parents[2] / ("teacher-cache-" + source)
        held = CachedTeacher(cache, "eval", device="cpu", max_length=1024,
            answer_marker=encode(ANSWER_MARKER), min_answer_tokens=2,
            strip_prefix=[(whole, 0, len(whole)), (heading, len(header), len(heading))],
            strip_nonthinking=(encode(ANSWER_MARKER), encode("\n<think>\n\n</think>")))
        selected = sorted(held.ids)[:64]
        ids = [held.read(d)["input_ids"][0].int() for d in selected]
        roles = [torch.from_numpy(roles_of(d.numpy())) for d in ids]
        domains[name], role_sets[name] = ids, roles
        counts = torch.cat([r[1:] for r in roles]).bincount(minlength=len(ROLES))
        report[name] = dict(documents=selected, eligible_eval_documents=len(held.ids),
            targets=int(counts.sum()), role_targets=dict(zip(ROLES, counts.tolist())),
            at_prefix_cap=sum(len(d) == 1024 for d in ids),
            tokenizer_eos=tok.eos_token_id)
        held.close()
    torch.save(dict(domains=domains, roles=role_sets, length=1024), out / "proxy-domains.pt")
    gsm, math = problems(256)
    banks = {"gsm8k": gsm, "math": math}
    # Match problem text as token subsequences, not just a reused source ID.
    needles = {name: [np.array(encode(q), dtype=np.int64) for q, _ in items] for name, items in banks.items()}
    plan = json.loads((HERE / "agentic-pilot-v2-balanced/plan.json").read_text())
    argv = plan["arms"]["agentic"]["argv"]
    start, stop = argv.index("--teacher-cache") + 1, argv.index("--assistant-only-caches")
    excluded = set(json.loads((HERE.parents[2] / "capture-data/exclude-long-r5.json").read_text()))
    index = defaultdict(list)
    for bank, arrays in needles.items():
        for i, a in enumerate(arrays):
            if len(a) >= 8:
                index[int(a[0])].append((bank, i, a))
    matches = []
    for path in argv[start:stop]:
        cache = OfflineTeacherCache(path)
        for doc in cache.document_ids("train"):
            if doc in excluded:
                continue
            ids = np.asarray(cache.read_document(doc, tokens_only=True)["input_ids"])
            for token in set(map(int, ids)) & index.keys():
                for at in np.flatnonzero(ids == token):
                    for bank, i, a in index[token]:
                        if at + len(a) <= len(ids) and np.array_equal(ids[at:at + len(a)], a):
                            matches.append(dict(bank=bank, index=i, cache=Path(path).name, doc_id=doc))
        cache.close()
        print("overlap checked " + Path(path).name, flush=True)
    report["math_overlap"] = dict(matches=matches, counts={bank: len({m["index"] for m in matches if m["bank"] == bank}) for bank in banks},
        method="exact complete problem token sequence against retained train documents; lower bound, not a paraphrase detector")
    (out / "proxy-audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "math-bank.json").write_text(json.dumps(banks, indent=2), encoding="utf-8")
    print(json.dumps({k: v if k == "math_overlap" else {a:b for a,b in v.items() if a != "documents"} for k,v in report.items()}, indent=2))


def targets(out, plan_path=None):
    from agentic_arm import recipe, teacher_for
    from training_state import PlannedBatches
    from teacher_kl import hedge_token_ids
    spec = json.loads((plan_path or HERE / "agentic-pilot-v2-balanced/plan.json").read_text())
    tok = AutoTokenizer.from_pretrained(HERE / "merges-long1/u50")
    totals, examples = {}, []
    for arm in ("agentic", "replay"):
        _, options = recipe(Path(spec["data"]), out / arm, spec["new_repeat"], spec["steps"], arm == "replay", **spec["tuning"])
        teacher = teacher_for(options)
        teacher.suppress = hedge_token_ids(tok)
        groups = teacher._groups(1, 128, 32768)
        count = spec['steps'] * 2
        batches = PlannedBatches(teacher, groups, 25, count if spec['tuning'].get('balanced') else 0)
        stats = defaultdict(lambda: defaultdict(float))
        for step in range(count):
            batch = batches.take(float("inf"))
            for row, doc in enumerate(batch["doc_ids"]):
                ids = batch["input_ids"][row].numpy()
                weights = batch["weight"][row].numpy() if "weight" in batch else np.r_[np.ones(len(ids)-1), 0]
                source = Path(teacher.cache._owner[doc][0]).name
                labels = roles_of(ids)[1:]
                w = weights[:-1]
                negative = batch.get("negative")
                negative = np.zeros(len(w), bool) if negative is None else negative[row, :-1].numpy()
                ce = 1. if batch["ce_only"] else (0. if batch["kl_only"] else .5)
                kl = 0. if batch["ce_only"] else (1. if batch["kl_only"] else .5)
                picks = batch["topk_ids"][row, :-1].numpy()
                probs = batch["topk_logprobs"][row, :-1].exp().numpy()
                argmax = probs.argmax(1)
                top = picks[np.arange(len(w)), argmax]
                true_mass = np.where(picks == ids[1:, None], probs, 0).sum(1)
                for role, label in enumerate(ROLES):
                    mask = labels == role
                    kweight = w * mask * ~negative * kl
                    s = stats[source + "/" + label]
                    s["targets"] += float((w * mask).sum())
                    s["ce_mass"] += float((w * mask * ce).sum())
                    s["kl_mass"] += float(kweight.sum())
                    s["ul_targets"] += float((w * mask * negative).sum())
                    s["teacher_top1_correct_mass"] += float((kweight * (top == ids[1:])).sum())
                    s["teacher_target_absent_mass"] += float((kweight * (true_mass == 0)).sum())
                    s["teacher_premature_stop_mass"] += float((kweight * np.isin(top, [248044,248046]) * ~np.isin(ids[1:],[248044,248046])).sum())
                if kl and len(examples) < 40:
                    own = np.isin(labels, [ROLES.index(x) for x in ("assistant", "thinking", "tool-call", "plain")])
                    take = np.flatnonzero((w > 0) & ~negative & own & (top != ids[1:]) & (probs.max(1) > .9))[:2]
                    for p in take:
                        examples.append(dict(arm=arm, source=source, doc_id=doc, role=ROLES[int(labels[p])],
                            prefix=tok.decode(ids[max(0,p-32):p+1]), actual=tok.decode([int(ids[p+1])]),
                            teacher=tok.decode([int(top[p])]), confidence=float(probs[p].max())))
            if (step + 1) % 10 == 0:
                print(f"{arm}: audited {step + 1}/{count} microbatches", flush=True)
        observed = sum(s['targets'] for s in stats.values())
        expected = spec['arms'][arm]['mixture']['prefix_total']
        if abs(observed - expected) > .01:
            raise ValueError(f"sampled targets differ from training plan: {observed} != {expected}")
        if spec['tuning'].get('mask_conversational'):
            raw = {'teacher-cache-frontier-code-raw', 'teacher-cache-general-pilot-w8'}
            context = sum(s['targets'] for key, s in stats.items()
                          if key.rsplit('/', 1)[0] not in raw
                          and key.rsplit('/', 1)[1] in ('system', 'user', 'tool-result'))
            if context:
                raise ValueError(f"conversational context still scored: {context}")
        totals[arm] = dict(stats)
        teacher.close()
        print("target audit complete " + arm, flush=True)
    (out / "replay-targets.json").write_text(json.dumps(dict(arms=totals, examples=examples), indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["proxies", "targets"])
    p.add_argument("--output", type=Path, default=HERE / "evaluation-audit")
    p.add_argument("--plan", type=Path, help="training plan to reconstruct in targets mode")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'targets':
        targets(args.output, args.plan)
    else:
        proxies(args.output)
