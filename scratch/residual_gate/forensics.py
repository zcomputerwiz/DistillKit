"""Take the trained checkpoints apart and ask which half carries the outcome.

Warm-starting the gate helped one seed by 0.0018 nats and hurt the other by 0.0042, with
no overlap between arms at either seed. Something in ``(B, G)`` decides that, and the
cheapest way to find out which is not another training grid -- it is to recombine the
states we already paid for and evaluate them.

Every combination is named on the command line as ``backbone+gate``, optionally with a
strength::

    C43+C43            the run as it was trained
    C43+S1             its backbone with the frozen-stage routing policy
    C43+none           its backbone with admission forced to 1
    C43+C43@0.5        its own policy at half strength
    C42+C43            the cross-seed swap

A gate is loaded into a backbone it was not trained with by installing a fresh gate of the
right family and loading the source state dict whole -- weights and frozen normalizer
together. Those two must travel as a pair: standardising one run's features against
another run's constants would be a third model neither arm ever trained, and would look
like a result.

No optimizer, no backward, no training. Evaluation semantics are the existing ones --
same bundle, same splits, same token classes, same baseline arithmetic.

    CUDA_VISIBLE_DEVICES=0 python scratch/residual_gate/forensics.py \
        --combo C43+C43 --combo C43+S1 --combo C43+none --output ...
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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

from attenuate import Familiarity, load_records
from coadapt_score import load_backbone, paired_difference, release
from distillkit.experimental.residual_gate import (
    TrigramFamiliarity, family_features, install_residual_gates,
    remove_residual_gates)
from evaluate import COUNT_EDGES, DEFAULT_CACHE, split_layout
from repeatability import DEFAULT_BUNDLE, trigram_keys

RUNS = Path("D:/DeepThought/Projects/HybridModel/runs")
STAGE1 = Path("scratch/residual_gate/gates/familiarity-lr3e3-step284.pt")
COMBO = re.compile(r"^([A-Za-z0-9_.-]+)\+([A-Za-z0-9_.-]+)(?:@([0-9.]+))?$")

# Every arm as (directory, seed, repeat). The manifest is built from this and from what
# is actually on disk, so a name that no longer resolves fails loudly.
ARMS = {
    "A42": "gate-coadapt-armA", "A42r1": "gate-coadapt-armA-r1",
    "A42r2": "gate-coadapt-armA-r2",
    "B42": "gate-coadapt-armB",
    "C42": "gate-coadapt-armC", "C42r1": "gate-coadapt-armC-r1",
    "C42r2": "gate-coadapt-armC-r2",
    "A43": "gate-coadapt-s43-armA", "A43m": "gate-coadapt-s43m-armA",
    "A43r": "gate-coadapt-s43r-armA",
    "B43": "gate-coadapt-s43-armB", "B43m": "gate-coadapt-s43m-armB",
    "B43r": "gate-coadapt-s43r-armB",
    "C43": "gate-coadapt-s43-armC", "C43r1": "gate-coadapt-s43-armC-r1",
    "C43r2": "gate-coadapt-s43-armC-r2",
}


def digest(path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def manifest(step: int) -> dict:
    entries = {}
    for name, folder in sorted(ARMS.items()):
        run = RUNS / folder
        checkpoints = sorted(int(path.name.split("-")[1]) for path in run.glob("checkpoint-*")
                             if path.is_dir())
        gates = sorted(int(path.stem.split("step")[1]) for path in run.glob("gate-step*.pt"))
        entries[name] = {
            "path": str(run),
            "arm": name[0], "seed": 42 if "42" in name else 43,
            "repeat": name[3:] or "orig",
            "checkpoints": checkpoints, "gates": gates,
            "has_gate": bool(gates),
            "has_optimizer_state": (run / ("checkpoint-%d" % step) / "optimizer.pt").exists(),
        }
    entries["S1"] = {"path": str(STAGE1), "arm": "stage1", "seed": None,
                     "repeat": "orig", "checkpoints": [], "gates": [284],
                     "has_gate": True, "has_optimizer_state": False,
                     "sha256": digest(STAGE1)}
    return entries


def gate_payload(name: str, step: int):
    if name == "none":
        return None
    if name == "S1":
        return torch.load(STAGE1, map_location="cpu", weights_only=False)
    run = RUNS / ARMS[name]
    path = run / ("gate-step%d.pt" % step)
    if not path.exists():
        raise SystemExit("no gate at step %d for %s" % (step, name))
    return torch.load(path, map_location="cpu", weights_only=False)


def bucket_statistics(rows, familiarity, vocab_size):
    """What the layer actually contributes, per familiarity bucket.

    The network consumes ``g * r``, not ``g``. Reporting the gate alone cannot
    distinguish a run that routes differently from a run whose FFN moved the other way
    and left the product where it was.
    """
    out = {}
    for layer in sorted({layer for _, kept, _ in rows for layer in kept}):
        gates, updates, counts = [], [], []
        for ids, kept, kept_norm in rows:
            if layer not in kept:
                continue
            gates.append(kept[layer].numpy())
            updates.append(kept_norm[layer].numpy())
            keys = trigram_keys(ids, vocab_size)
            counts.append(np.array(
                [familiarity.counts.get(key, 0) if key is not None else 0
                 for key in keys], dtype=np.float64))
        gate = np.concatenate(gates)
        update = np.concatenate(updates)
        count = np.concatenate(counts)
        entry = {}
        edges = list(zip(COUNT_EDGES[:-1], COUNT_EDGES[1:]))
        edges.append((COUNT_EDGES[-1], float("inf")))
        for low, high in edges:
            inside = (count >= low) & (count < high)
            label = "%d-%d" % (low, high) if high != float("inf") else "%d+" % low
            if not inside.any():
                entry[label] = None
                continue
            chosen_gate = gate[inside]
            chosen_update = update[inside]
            entry[label] = {
                "g": float(chosen_gate.mean()),
                "g_median": float(np.median(chosen_gate)),
                "g_p25": float(np.percentile(chosen_gate, 25)),
                "g_p75": float(np.percentile(chosen_gate, 75)),
                "r": float(chosen_update.mean()),
                "gr": float((chosen_gate * chosen_update).mean()),
                "delta_r": float((np.abs(chosen_gate - 1.0) * chosen_update).mean()),
            }
        out[str(layer)] = entry
    return out


@torch.inference_mode()
def score(model, records, device, classes, handle, familiarity, vocab_size, collect):
    """Per-class per-document NLL, and -- when asked -- what the gated layers did.

    The gate values and the update norms have to be copied out inside the same document
    loop: the handle is reset per document, so anything read afterwards describes only
    the last one.
    """
    out = {}
    gathered = []
    for record in records:
        ids = record["ids"]
        tokens = torch.tensor([ids], device=device)
        if handle is not None:
            handle.reset_stats()
        logits = model(input_ids=tokens,
                       attention_mask=torch.ones_like(tokens)).logits[0, :-1].float()
        targets = ids[1:]
        nll = torch.nn.functional.cross_entropy(
            logits, torch.tensor(targets, device=device), reduction="none").cpu()
        if collect and handle is not None:
            gathered.append((ids, {layer: torch.cat([p.reshape(-1) for p in parts])
                                   for layer, parts in handle.kept.items()},
                             {layer: torch.cat([p.reshape(-1) for p in parts])
                              for layer, parts in handle.kept_norms.items()}))
        sums = {}
        for index, token in enumerate(targets):
            entry = sums.setdefault(classes[token], [0.0, 0])
            entry[0] += float(nll[index])
            entry[1] += 1
        sums["all"] = [float(nll.sum()), len(targets)]
        for label, (value, count) in sums.items():
            out.setdefault(label, []).append(value / count)
    grid = (bucket_statistics(gathered, familiarity, vocab_size)
            if collect and gathered else None)
    return out, grid


def evaluate_combo(backbone, gate_name, strength, step, records, classes, args,
                   familiarity, collect):
    """One (backbone, gate, strength) triple, scored on the current split."""
    checkpoint = RUNS / ARMS[backbone] / ("checkpoint-%d" % step)
    if not checkpoint.is_dir():
        raise SystemExit("no checkpoint-%d for backbone %s" % (step, backbone))
    model, config = load_backbone(checkpoint, args.device)
    handle = None
    payload = gate_payload(gate_name, step)
    if payload is not None:
        source = (TrigramFamiliarity(args.cache, config.vocab_size)
                  if "log_count" in family_features(payload["family"]) else None)
        handle = install_residual_gates(model, payload["layers"],
                                        family=payload["family"], familiarity=source)
        # Weights and normalizer together: loading one run's weights against another's
        # constants would be a model neither arm ever trained.
        model.residual_gates.load_state_dict(payload["state_dict"], strict=True)
        handle.strength = strength
        handle.keep = collect
        handle.keep_update = collect
    values, grid = score(model, records, args.device, classes, handle, familiarity,
                         config.vocab_size, collect)
    if handle is not None:
        remove_residual_gates(model)
    release(model)
    return values, grid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combo", action="append", required=True,
                        help="backbone+gate[@strength]; repeat")
    parser.add_argument("--reference", default=None,
                        help="combo every other combo is differenced against")
    parser.add_argument("--step", type=int, default=284)
    parser.add_argument("--tokenizer",
                        default="D:/DeepThought/Projects/HybridModel/student-2b-hf")
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--split", default="screen")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--statistics", action="store_true",
                        help="also record g, ||r|| and ||g r|| per layer and bucket")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise SystemExit("mask to one card with CUDA_VISIBLE_DEVICES")

    from transformers import AutoConfig, AutoTokenizer

    from distillkit.independent_eval import build_token_classes

    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    base = AutoConfig.from_pretrained(args.tokenizer, local_files_only=True)
    vocab_size = getattr(base, "text_config", base).vocab_size
    classes = split_layout(build_token_classes(tokenizer, vocab_size), tokenizer,
                           vocab_size)
    records = load_records(args.bundle, args.split, args.limit)
    familiarity = Familiarity(args.cache)

    report = {"split": args.split, "step": args.step, "documents": len(records),
              "combos": {}, "differences": {}}
    scored = {}
    for entry in args.combo:
        match = COMBO.match(entry)
        if not match:
            raise SystemExit("expected backbone+gate[@strength], got %r" % entry)
        backbone, gate_name, strength = match.group(1), match.group(2), match.group(3)
        strength = 1.0 if strength is None else float(strength)
        values, grid = evaluate_combo(backbone, gate_name, strength, args.step, records,
                                      classes, args, familiarity, args.statistics)
        scored[entry] = values
        report["combos"][entry] = {
            "backbone": backbone, "gate": gate_name, "strength": strength,
            "nll": {label: statistics.fmean(series)
                    for label, series in sorted(values.items())},
            # Kept so any pair of combos can be differenced afterwards without another
            # forward pass; a reference chosen at launch should not decide what the
            # analysis is allowed to ask.
            "per_document": {label: series for label, series in sorted(values.items())},
        }
        if grid is not None:
            report["combos"][entry]["update"] = grid
        print("%-22s content %.6f" % (entry, report["combos"][entry]["nll"]["content"]),
              flush=True)

    reference = args.reference or args.combo[0]
    for entry in args.combo:
        if entry == reference:
            continue
        report["differences"]["%s - %s" % (entry, reference)] = paired_difference(
            scored[entry], scored[reference])

    report["manifest"] = manifest(args.step)
    report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote %s in %.0f s" % (args.output, report["elapsed_seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
