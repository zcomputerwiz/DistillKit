"""Score several co-adaptation arms at shared milestones, and pair every difference.

The fresh-gate experiment needed two arms. The warm-start experiment needs three -- stock,
fresh-gate, and a gate started from the policy it learned against the frozen backbone --
plus the forced-identity variant of each gated arm, all paired by document on the same
corpus. So arms are named on the command line and the differences that matter are
reported::

    --arm B=.../gate-coadapt-armB --arm A=.../gate-coadapt-armA --arm C=.../gate-coadapt-armC

An arm is gated when a ``gate-step<N>.pt`` sits beside its checkpoint, and each gated arm
is scored twice: once normally and once with admission forced back to unit strength. That
second reading is what separates "the gate routes better" from "the backbone reorganised
differently", and the two must never be quoted as the same quantity -- ``C - A`` is the
initialisation comparison, ``C - C(g=1)`` is a statement about how much this particular
model leans on its gate at inference.

One arm at a time, per-document class means retained, so pairing is exact without three
2B models resident at once. Gate admission is also recorded per layer and familiarity
bucket: gate weights are not comparable between runs -- a permuted hidden layer is the
same function -- but what the gate *does* to a given context at a given depth is, and
that grid is what the cross-seed stability question is properly asked of.

    CUDA_VISIBLE_DEVICES=0 python scratch/residual_gate/coadapt_score.py \
        --arm B=... --arm A=... --arm C=... --output scratch/residual_gate/warm.json
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffn_memo"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffn_attenuation"))

import numpy as np
import torch
import torch.nn.functional as F

from attenuate import Familiarity, load_records
from distillkit.experimental.residual_gate import (
    TrigramFamiliarity, family_features, install_residual_gates,
    remove_residual_gates)
from evaluate import COUNT_EDGES, DEFAULT_CACHE, split_layout
from repeatability import DEFAULT_BUNDLE, trigram_keys

CHECKPOINT = re.compile(r"checkpoint-(\d+)$")


def milestones(run: Path):
    found = []
    for path in run.iterdir():
        match = CHECKPOINT.search(path.name)
        if match and path.is_dir():
            found.append((int(match.group(1)), path))
    if not found:
        raise SystemExit("no checkpoints under %s" % run)
    return dict(sorted(found))


def paired_difference(left, right):
    """Per-document left - right, for every class both arms scored."""
    out = {}
    for label in sorted(set(left) & set(right)):
        values = [a - b for a, b in zip(left[label], right[label])]
        mean = statistics.fmean(values)
        error = (statistics.stdev(values) / len(values) ** 0.5
                 if len(values) > 1 else 0.0)
        out[label] = {"mean": mean, "stderr": error,
                      "t": mean / error if error else 0.0,
                      "ci95": [mean - 1.96 * error, mean + 1.96 * error],
                      "better": sum(1 for value in values if value < 0),
                      "n": len(values)}
    return out


@torch.inference_mode()
def per_document(model, records, device, classes, handle=None):
    """Mean NLL per class per document, and the gate values behind it."""
    out = {}
    rows = []
    for record in records:
        ids = record["ids"]
        tokens = torch.tensor([ids], device=device)
        if handle is not None:
            handle.reset_stats()
        logits = model(input_ids=tokens,
                       attention_mask=torch.ones_like(tokens)).logits[0, :-1].float()
        targets = ids[1:]
        nll = F.cross_entropy(logits, torch.tensor(targets, device=device),
                              reduction="none").cpu()
        if handle is not None and handle.keep:
            rows.append((ids, handle.kept))
        sums = {}
        for index, token in enumerate(targets):
            entry = sums.setdefault(classes[token], [0.0, 0])
            entry[0] += float(nll[index])
            entry[1] += 1
        sums["all"] = [float(nll.sum()), len(targets)]
        for label, (value, count) in sums.items():
            out.setdefault(label, []).append(value / count)
    return out, rows


def policy_grid(rows, familiarity, vocab_size):
    """Mean admission per gated layer per familiarity bucket: the comparable policy."""
    grid = {}
    for layer in sorted({layer for _, kept in rows for layer in kept}):
        values, counts = [], []
        for ids, kept in rows:
            if layer not in kept:
                continue
            gates = torch.cat([part.reshape(-1) for part in kept[layer]]).numpy()
            keys = trigram_keys(ids, vocab_size)
            if len(gates) != len(ids):
                raise ValueError("kept %d gate values for %d tokens"
                                 % (len(gates), len(ids)))
            values.append(gates)
            counts.append(np.array(
                [familiarity.counts.get(key, 0) if key is not None else 0
                 for key in keys], dtype=np.float64))
        values = np.concatenate(values)
        counts = np.concatenate(counts)
        buckets = {}
        for low, high in zip(COUNT_EDGES[:-1], COUNT_EDGES[1:]):
            inside = (counts >= low) & (counts < high)
            buckets["%d-%d" % (low, high)] = (float(values[inside].mean())
                                              if inside.any() else None)
        inside = counts >= COUNT_EDGES[-1]
        buckets["%d+" % COUNT_EDGES[-1]] = (float(values[inside].mean())
                                            if inside.any() else None)
        buckets["all"] = float(values.mean())
        # The largest departure from unit admission actually observed, which is the
        # comparable version of the training log's parameter-space `reach`.
        buckets["reach"] = float(np.abs(values - 1.0).max())
        grid[str(layer)] = buckets
    return grid


def grid_similarity(left, right):
    """How alike two routing policies are over the layer x familiarity cells."""
    pairs = []
    for layer in sorted(set(left) & set(right)):
        for bucket in sorted(set(left[layer]) & set(right[layer])):
            if bucket in ("all", "reach"):
                continue
            first, second = left[layer][bucket], right[layer][bucket]
            if first is not None and second is not None:
                pairs.append((first, second))
    if len(pairs) < 3:
        return None
    first = np.array([pair[0] for pair in pairs])
    second = np.array([pair[1] for pair in pairs])
    difference = first - second
    correlation = (float(np.corrcoef(first, second)[0, 1])
                   if first.std() and second.std() else None)
    return {"cells": len(pairs), "pearson": correlation,
            "rms": float(np.sqrt((difference ** 2).mean())),
            "mean_absolute": float(np.abs(difference).mean())}


def load_backbone(path, device):
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    config = AutoConfig.from_pretrained(path, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.use_cache = False
    model = Qwen3_5ForCausalLM.from_pretrained(
        path, config=config, dtype=torch.bfloat16,
        local_files_only=True).to(device).eval()
    return model, config


def release(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def gate_drift(payload, reference):
    """Parameter displacement from the warm start, reported beside functional drift.

    Never instead of it: two gates with the same weights compute the same function, but
    two gates computing the same function need not have the same weights.
    """
    if reference is None:
        return None
    moved, base = 0.0, 0.0
    for key, value in payload["state_dict"].items():
        if not key.endswith(("weight", "bias")):
            continue
        start = reference["state_dict"][key].float()
        moved += float(((value.float() - start) ** 2).sum())
        base += float((start ** 2).sum())
    return {"l2": moved ** 0.5, "relative": (moved ** 0.5) / (base ** 0.5 + 1e-12)}


def wanted_differences(names, reference):
    """Every gated arm against the reference and against each other, plus its own g=1."""
    plain = [name for name in names if not name.endswith("(g=1)")]
    pairs = []
    for index, left in enumerate(plain):
        if left != reference:
            pairs.append((left, reference))
        for right in plain[:index]:
            if right != reference:
                pairs.append((left, right))
        if left + "(g=1)" in names:
            pairs.append((left, left + "(g=1)"))
    for left in plain:
        identity = left + "(g=1)"
        if identity in names and reference in names:
            pairs.append((identity, reference))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True,
                        help="NAME=PATH; repeat. The first is the difference reference.")
    parser.add_argument("--tokenizer",
                        default="D:/DeepThought/Projects/HybridModel/student-2b-hf")
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--split", default="screen")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--warm-start-gate", type=Path, default=None,
                        help="the frozen-stage gate a warm-started arm began from")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise SystemExit("mask to one card with CUDA_VISIBLE_DEVICES")

    from transformers import AutoConfig, AutoTokenizer

    from distillkit.independent_eval import build_token_classes

    started = time.monotonic()
    arms = {}
    for entry in args.arm:
        name, _, path = entry.partition("=")
        if not path:
            raise SystemExit("expected NAME=PATH, got %r" % entry)
        arms[name] = Path(path)
    reference = next(iter(arms))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    base = AutoConfig.from_pretrained(args.tokenizer, local_files_only=True)
    vocab_size = getattr(base, "text_config", base).vocab_size
    classes = split_layout(build_token_classes(tokenizer, vocab_size), tokenizer,
                           vocab_size)
    records = load_records(args.bundle, args.split, args.limit)
    familiarity = Familiarity(args.cache)
    warm_start = (torch.load(args.warm_start_gate, map_location="cpu",
                             weights_only=False)
                  if args.warm_start_gate else None)

    steps = {name: milestones(path) for name, path in arms.items()}
    shared = sorted(set.intersection(*(set(value) for value in steps.values())))
    if not shared:
        raise SystemExit("the arms share no milestone; they are not comparable")

    report = {"arms": {name: str(path) for name, path in arms.items()},
              "reference": reference, "split": args.split,
              "documents": len(records), "shared_milestones": shared,
              "milestones": []}
    if warm_start is not None:
        report["warm_start_gate"] = str(args.warm_start_gate)

    for step in shared:
        scored, policies, drifts = {}, {}, {}
        for name, run in arms.items():
            model, config = load_backbone(steps[name][step], args.device)
            gate_path = run / ("gate-step%d.pt" % step)
            handle = None
            if gate_path.exists():
                payload = torch.load(gate_path, map_location="cpu", weights_only=False)
                source = (TrigramFamiliarity(args.cache, config.vocab_size)
                          if "log_count" in family_features(payload["family"]) else None)
                handle = install_residual_gates(model, payload["layers"],
                                                family=payload["family"],
                                                familiarity=source)
                model.residual_gates.load_state_dict(payload["state_dict"], strict=True)
                handle.keep = True
                drift = gate_drift(payload, warm_start)
                if drift is not None:
                    drifts[name] = drift
            values, rows = per_document(model, records, args.device, classes, handle)
            scored[name] = values
            if handle is not None:
                handle.keep = False
                policies[name] = policy_grid(rows, familiarity, config.vocab_size)
                handle.force_identity = True
                identity, _ = per_document(model, records, args.device, classes, handle)
                scored[name + "(g=1)"] = identity
                remove_residual_gates(model)
            release(model)

        entry = {"step": step,
                 "content_nll": {name: statistics.fmean(values["content"])
                                 for name, values in scored.items()},
                 "differences": {}, "policy": policies}
        if drifts:
            entry["gate_parameter_drift"] = drifts
        for left, right in wanted_differences(list(scored), reference):
            entry["differences"]["%s - %s" % (left, right)] = paired_difference(
                scored[left], scored[right])
        report["milestones"].append(entry)
        print("step %-4d %s" % (step, "  ".join(
            "%s %+.6f" % (label, value["content"]["mean"])
            for label, value in entry["differences"].items())), flush=True)

    final = report["milestones"][-1]["policy"]
    report["policy_similarity"] = {
        "%s vs %s" % (left, right): grid_similarity(final[left], final[right])
        for left in sorted(final) for right in sorted(final) if left < right}
    report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote %s in %.0f s" % (args.output, report["elapsed_seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
