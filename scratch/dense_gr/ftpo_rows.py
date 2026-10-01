"""FTPO rows (after Liquid4All/antidoom) from looping rollouts: one decision per loop.

For each rollout that loops (`teacher_kl.loop_start`), the rejected token is the first
readable token of the copy -- whitespace and punctuation begin legitimate text as often as
loops -- and the chosen tokens are the reference's own alternatives there: up to
--max-chosen with probability at least --min-chosen-p, readable or the thought's close,
never a hedge opener and never the rejected word respelled. A row is kept only when the
reference itself would likely emit the rejected token (--min-rejected-p), so every
negative is one the model being trained plausibly produces, whichever checkpoint wrote
the rollout. The reference's top-k logits ride along for the trainer's MSE tether.

Frequency is flattened as antidoom does: a row whose rejected token is common is kept
with probability (median count / count) ** --rejected-strength, and each chosen token
likewise ** --chosen-strength, never emptying a row.

    python scratch/dense_gr/ftpo_rows.py --reference <ckpt> \\
        --inputs ../capture-data/onpolicy-r8-g0.jsonl --output ../capture-data/ftpo-r9.jsonl
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from teacher_kl import HEDGE_OPENERS, loop_start  # noqa: E402

# A word: digits restate the problem's numbers ("shoot 4 times"), where an alternative
# changes content rather than breaks a loop; "<|im_end|>" spells letters but is no word.
READABLE = re.compile(r"[A-Za-z]")


def first_readable(ids, at, readable):
    while at < len(ids) and not readable(ids[at]):
        at += 1
    return at if at < len(ids) else None


def alternatives(probs, rejected, *, spell, readable, allowed, top, min_p, limit):
    """Chosen token ids, most probable first."""
    word = spell(rejected).strip().lower()
    out = []
    for token in np.argsort(-probs)[:top]:
        token = int(token)
        if probs[token] < min_p or len(out) == limit:
            break
        if token == rejected or spell(token).strip() in HEDGE_OPENERS:
            continue
        if (readable(token) or token in allowed) and spell(token).strip().lower() != word:
            out.append(token)
    return out


def flatten(rows, rejected_strength, chosen_strength, seed=0):
    """antidoom's frequency regularisation: cull rows by rejected token, prune chosen."""
    rng = np.random.default_rng(seed)
    keep = lambda counts, token, s: rng.random() < min(1.0, (np.median(list(counts.values())) / counts[token]) ** s)
    counts = Counter(r["rejected"] for r in rows)
    rows = [r for r in rows if keep(counts, r["rejected"], rejected_strength)]
    counts = Counter(t for r in rows for t in r["chosen"])
    for r in rows:
        kept = [t for t in r["chosen"] if keep(counts, t, chosen_strength)]
        r["chosen"] = kept or r["chosen"][:1]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, default=None, help="JSON list of document ids to skip")
    parser.add_argument("--min-rejected-p", type=float, default=0.3)
    parser.add_argument("--min-chosen-p", type=float, default=0.02)
    parser.add_argument("--max-chosen", type=int, default=3)
    parser.add_argument("--top", type=int, default=512, help="reference logits kept for the tether")
    parser.add_argument("--max-prefix", type=int, default=2048)
    parser.add_argument("--rejected-strength", type=float, default=0.3)
    parser.add_argument("--chosen-strength", type=float, default=0.5)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    import torch
    from transformers import AutoTokenizer

    from distillkit.models import Qwen35WidenedForCausalLM

    tok = AutoTokenizer.from_pretrained(args.reference)
    model = Qwen35WidenedForCausalLM.from_pretrained(args.reference, dtype=torch.bfloat16).cuda().eval()
    encode = lambda text: tok(text, add_special_tokens=False)["input_ids"]
    spelled = {}
    spell = lambda t: spelled.setdefault(t, tok.decode([t]))
    readable = lambda t: bool(READABLE.search(spell(t))) and "<|" not in spell(t)
    close = tok.convert_tokens_to_ids("</think>")
    excluded = set(json.load(open(args.exclude))) if args.exclude else set()
    rows, seen, decisions = [], Counter(), set()
    for path in args.inputs:
        for line in open(path, encoding="utf-8"):
            rollout = json.loads(line)
            # A rollout's id is its prompt document's, wrapped: onpolicy:<doc>:<s0|greedy>.
            if {rollout["doc_id"], re.sub(r"^onpolicy:|:(s\d+|greedy)$", "", rollout["doc_id"])} & excluded:
                seen["excluded"] += 1
                continue
            seen["rollouts"] += 1
            text, cut = rollout["text"], rollout["prompt_chars"]
            prompt = encode(text[:cut])
            ids = prompt + encode(text[cut:])
            at = loop_start(ids, len(prompt), close)
            if at is None:
                continue
            seen["looping"] += 1
            at = first_readable(ids, at, readable)
            if at is None or at > args.max_prefix:
                seen["no readable start within the cap"] += 1
                continue
            key = hash(tuple(ids[:at + 1]))
            if key in decisions:  # the same greedy rollout in two pools
                seen["duplicate"] += 1
                continue
            decisions.add(key)
            prefix = torch.tensor([ids[:at]], device="cuda")
            with torch.no_grad():
                hidden = model.model(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                                     use_cache=False).last_hidden_state
                logits = model.lm_head(hidden[0, -1]).float()
            probs = torch.softmax(logits, -1).cpu().numpy()
            rejected = ids[at]
            if probs[rejected] < args.min_rejected_p:
                seen["rejected implausible for the reference"] += 1
                continue
            chosen = alternatives(probs, rejected, spell=spell, readable=readable, allowed={close},
                                  top=args.top, min_p=args.min_chosen_p, limit=args.max_chosen)
            if not chosen:
                seen["no alternative"] += 1
                continue
            top = torch.topk(logits, args.top)
            rows.append({"doc_id": rollout["doc_id"], "source": rollout.get("source"),
                         "prefix_ids": ids[:at], "rejected": rejected, "chosen": chosen,
                         "p_rejected": round(float(probs[rejected]), 4),
                         "ref_ids": top.indices.tolist(),
                         "ref_logits": [round(v, 3) for v in top.values.tolist()],
                         "targets": {t: round(float(logits[t]), 3) for t in chosen + [rejected]}})
    rows = flatten(rows, args.rejected_strength, args.chosen_strength)
    with open(args.output, "w", encoding="utf-8") as out:
        for row in rows:
            targets = row.pop("targets")
            row["ref_target_logits"] = [targets[t] for t in row["chosen"] + [row["rejected"]]]
            out.write(json.dumps(row) + "\n")
    rejected = Counter(spell(r["rejected"]) for r in rows).most_common(8)
    chosen = Counter(spell(t) for r in rows for t in r["chosen"]).most_common(8)
    print("%s; %d rows -> %s" % (dict(seen), len(rows), args.output))
    print("most rejected %s\nmost chosen %s" % (rejected, chosen))


if __name__ == "__main__":
    main()
