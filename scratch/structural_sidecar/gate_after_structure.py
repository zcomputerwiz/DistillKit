"""Does solving structure change what the residual gate wants to learn?

The residual gate was fitted before the structural sidecar existed, so its 260 parameters
were spent against an error budget that still contained half a nat of newline and a fifth
of a nat of punctuation. The sidecar now removes most of that without touching the
backbone. If the gate's remaining pressure is then mostly content, a gate refitted in the
sidecar's presence should find a different -- and more content-directed -- policy.

That is a claim about what the optimizer sees, so it is measured before anything is
trained: per-class gate-gradient norms at exact identity, with the sidecar off and on. If
the structural share of the gradient does not move, the hypothesis is already weak and the
training that follows is a formality.

Everything except the new gate is frozen and hashed before and after. A second gate is
also fitted in this same harness *without* the sidecar, so a difference between the old
gate and the new one cannot be blamed on the harness rather than on the sidecar.

    CUDA_VISIBLE_DEVICES=0 python scratch/structural_sidecar/gate_after_structure.py \
        --output scratch/structural_sidecar/gate-after-S
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffn_memo"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffn_attenuation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "residual_gate"))

import torch
import torch.nn.functional as F

from attenuate import load_records
from distillkit.experimental.ngram_hash import NGramHashConfig, NGramHasher
from distillkit.experimental.residual_gate import (
    TrigramFamiliarity, calibrate_gates, install_residual_gates,
    remove_residual_gates)
from distillkit.experimental.structural_sidecar import (
    FactorizedSidecar, StructuralSidecar, apply_structural_bias,
    structural_token_ids)
from evaluate import DEFAULT_CACHE, split_layout
from factorize import class_ids
from fit import BACKBONES, BASE, corpus
from repeatability import DEFAULT_BUNDLE, held_out_digests

GATE = Path("scratch/residual_gate/gates/familiarity-lr3e3-step284.pt")
ADDRESSED = Path("scratch/structural_sidecar/direct-B42/sidecar.pt")
WHITESPACE = Path("scratch/structural_sidecar/factorized-B42-stockref/whitespace.pt")
CLASSES = ("content", "newline", "whitespace", "punctuation", "control")


def digest_of(module, skip: str = "residual_gates.") -> str:
    """Hash the parameters, ignoring anything the run is allowed to train.

    Installing the gate makes it a submodule of the model, so a naive hash over
    named_parameters would report the frozen backbone as changed the moment the gate
    took its first step -- which is the check succeeding by accident and then failing
    loudly for the wrong reason.
    """
    hasher = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        if skip and name.startswith(skip):
            continue
        hasher.update(name.encode("utf-8"))
        hasher.update(parameter.detach().to(torch.float32).cpu().numpy().tobytes())
    for name, buffer in sorted(module.named_buffers()):
        if skip and name.startswith(skip):
            continue
        hasher.update(name.encode("utf-8"))
        hasher.update(buffer.detach().to(torch.float32).cpu().numpy().tobytes())
    return hasher.hexdigest()


def build_structural(config, structural, whitespace, device):
    """The canonical sidecar at its selected operating point, frozen."""
    saved = torch.load(ADDRESSED, map_location="cpu", weights_only=False)
    addressed = StructuralSidecar(rows=saved["rows"], code_dim=saved["code_dim"],
                                  structural=int(structural.numel()), mode=saved["mode"],
                                  heads=saved["heads"], hidden=saved["hidden"],
                                  seed=saved["seed"]).to(device).to(torch.float32)
    addressed.load_state_dict(saved["state_dict"])
    module = FactorizedSidecar(addressed, structural.cpu(), whitespace.cpu()).to(device)
    bias = torch.load(WHITESPACE, map_location=device, weights_only=False)
    module.white.load_state_dict(bias["state_dict"])
    module.requires_grad_(False)
    hasher = NGramHasher(NGramHashConfig(
        vocab_size=config.vocab_size, ngram_size=3, heads_per_ngram=1,
        ngram_vocab_size_base=saved["rows"] // 2 - 64, seed=1234,
        eos_token_id=config.eos_token_id if hasattr(config, "eos_token_id")
        else config.vocab_size - 1))
    return module, hasher, bias["whitespace_strength"]


def corrected(model, ids, device, structural, sidecar, hasher, strength):
    """Logits with the frozen structural correction applied, or without it."""
    tokens = torch.tensor([ids], device=device)
    logits = model(input_ids=tokens,
                   attention_mask=torch.ones_like(tokens)).logits[0, :-1].float()
    if sidecar is None:
        return logits
    bias = sidecar(hasher.row_indices(tokens), strength)[0, :-1]
    return apply_structural_bias(logits.unsqueeze(0), bias.unsqueeze(0), structural,
                                 1.0)[0]


def class_gradients(model, handle, documents, device, classes, structural, sidecar,
                    hasher, strength):
    """Per-class gate gradient at identity: what the optimizer is actually pushed by."""
    parameters = [p for p in handle.gates.parameters() if p.requires_grad]
    sums = {label: [torch.zeros_like(p) for p in parameters] for label in CLASSES}
    counts = dict.fromkeys(CLASSES, 0)
    for ids in documents:
        logits = corrected(model, ids, device, structural, sidecar, hasher, strength)
        targets = torch.tensor(ids[1:], device=device)
        nll = F.cross_entropy(logits, targets, reduction="none")
        labels = [classes[token] for token in ids[1:]]
        for label in CLASSES:
            mask = torch.tensor([value == label for value in labels], device=device)
            if not mask.any():
                continue
            for parameter in parameters:
                parameter.grad = None
            nll[mask].mean().backward(retain_graph=True)
            for index, parameter in enumerate(parameters):
                if parameter.grad is not None:
                    sums[label][index] += parameter.grad.detach().clone()
            counts[label] += 1
    flat = {}
    for label in CLASSES:
        if not counts[label]:
            continue
        flat[label] = torch.cat([(value / counts[label]).reshape(-1)
                                 for value in sums[label]])
    total = sum(float(value.norm()) for value in flat.values()) or 1.0
    report = {"norms": {label: float(value.norm()) for label, value in flat.items()},
              "fraction": {label: float(value.norm()) / total
                           for label, value in flat.items()},
              "cosine": {}}
    for label in CLASSES:
        if label == "content" or label not in flat or "content" not in flat:
            continue
        report["cosine"]["content vs %s" % label] = float(
            F.cosine_similarity(flat["content"].unsqueeze(0),
                                flat[label].unsqueeze(0)).item())
    return report


@torch.no_grad()
def evaluate(model, documents, device, classes, structural, sidecar, hasher, strength):
    per_document, totals = {}, {}
    for ids in documents:
        logits = corrected(model, ids, device, structural, sidecar, hasher, strength)
        targets = torch.tensor(ids[1:], device=device)
        nll = F.cross_entropy(logits, targets, reduction="none").cpu()
        sums = {}
        for index, token in enumerate(ids[1:]):
            label = classes[token]
            bucket = totals.setdefault(label, [0.0, 0])
            bucket[0] += float(nll[index])
            bucket[1] += 1
            entry = sums.setdefault(label, [0.0, 0])
            entry[0] += float(nll[index])
            entry[1] += 1
        sums["all"] = [float(nll.sum()), len(ids) - 1]
        totals.setdefault("all", [0.0, 0])
        totals["all"][0] += float(nll.sum())
        totals["all"][1] += len(ids) - 1
        for label, (value, count) in sums.items():
            per_document.setdefault(label, []).append(value / count)
    return ({label: statistics.fmean(values) for label, values in per_document.items()},
            {label: {"nll": bucket[0] / bucket[1], "tokens": bucket[1]}
             for label, bucket in sorted(totals.items())})


def install_fresh_gate(model, familiarity_cache, vocab_size, device, documents):
    payload = torch.load(GATE, map_location="cpu", weights_only=False)
    statistics_source = TrigramFamiliarity(familiarity_cache, vocab_size)
    handle = install_residual_gates(model, payload["layers"], family=payload["family"],
                                    familiarity=statistics_source)
    batches = []
    for ids in documents[:8]:
        tokens = torch.tensor([ids], device=device)
        batches.append({"input_ids": tokens, "attention_mask": torch.ones_like(tokens)})
    calibrate_gates(model, handle, batches)
    return handle, payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="B42", choices=sorted(BACKBONES))
    parser.add_argument("--no-structure", action="store_true",
                        help="fit the control gate in this same harness, sidecar off")
    parser.add_argument("--documents", type=int, default=512)
    parser.add_argument("--gradient-documents", type=int, default=48)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise SystemExit("mask to one card with CUDA_VISIBLE_DEVICES")

    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from distillkit.independent_eval import build_token_classes

    started = time.monotonic()
    checkpoint = BACKBONES[args.backbone]
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.use_cache = False
    model = Qwen3_5ForCausalLM.from_pretrained(
        checkpoint, config=config, dtype=torch.bfloat16,
        local_files_only=True).to(args.device).eval()
    model.requires_grad_(False)
    backbone_digest = digest_of(model)

    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    classes = split_layout(build_token_classes(tokenizer, config.vocab_size), tokenizer,
                           config.vocab_size)
    structural = structural_token_ids(classes, device=args.device)
    whitespace = class_ids(classes, "whitespace", device=args.device)
    sidecar, hasher, strength = build_structural(config, structural, whitespace,
                                                 args.device)
    sidecar_digest = digest_of(sidecar, skip="")
    if args.no_structure:
        sidecar = None

    excluded = held_out_digests(args.bundle)
    train = corpus(tokenizer, excluded, args.documents, args.length, skip=0)
    handle, payload = install_fresh_gate(model, args.cache, config.vocab_size,
                                         args.device, train)
    report = {"backbone": args.backbone, "structure": not args.no_structure,
              "gate_layers": payload["layers"], "gate_family": payload["family"],
              "parameters": sum(p.numel() for p in handle.gates.parameters()),
              "backbone_sha256": backbone_digest,
              "structural_sha256": sidecar_digest}
    print(json.dumps({k: report[k] for k in ("structure", "parameters", "gate_layers")}),
          flush=True)

    # The diagnostic Codex asked not to skip: what the gate is pushed by, at identity,
    # with the structural correction off and on. Both are measured here whatever this
    # run trains, so one process produces the comparison.
    report["gradients"] = {}
    for label, applied in (("without_structure", None), ("with_structure", sidecar)):
        if label == "with_structure" and args.no_structure:
            continue
        report["gradients"][label] = class_gradients(
            model, handle, train[:args.gradient_documents], args.device, classes,
            structural, applied, hasher, strength)
        entry = report["gradients"][label]
        print("%-18s %s" % (label, "  ".join(
            "%s %.3e (%.1f%%)" % (name, entry["norms"][name],
                                  100 * entry["fraction"][name])
            for name in CLASSES if name in entry["norms"])), flush=True)
        print("%-18s %s" % ("", "  ".join(
            "%s %+.3f" % (key.replace("content vs ", ""), value)
            for key, value in entry["cosine"].items())), flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in handle.gates.parameters() if p.requires_grad], lr=args.lr)
    history = []
    for epoch in range(args.epochs):
        losses = []
        for ids in train:
            logits = corrected(model, ids, args.device, structural, sidecar, hasher,
                               strength)
            targets = torch.tensor(ids[1:], device=args.device)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(statistics.fmean(losses))
        print("epoch %d loss %.5f" % (epoch, history[-1]), flush=True)
    report["loss_history"] = history
    report["reach"] = {str(index): handle.gate(index).gate_report("g")["g/reach"]
                       for index in handle.layer_indices}
    print("reach %s" % json.dumps(report["reach"]), flush=True)

    for split in ("screen", "confirmation"):
        documents = [record["ids"] for record in load_records(args.bundle, split, 0)]
        entry = {}
        for name, gate_on, structure_on in (("stock", False, False),
                                            ("gate", True, False),
                                            ("structure", False, True),
                                            ("gate+structure", True, True)):
            handle.force_identity = not gate_on
            applied = sidecar if structure_on and not args.no_structure else None
            means, totals = evaluate(model, documents, args.device, classes, structural,
                                     applied, hasher, strength)
            entry[name] = {"per_document_mean": means, "totals": totals}
        handle.force_identity = False
        report[split] = entry
        base = entry["stock"]["per_document_mean"]
        for name in ("gate", "structure", "gate+structure"):
            means = entry[name]["per_document_mean"]
            print("%-12s %-16s %s" % (split, name, "  ".join(
                "%s %+.6f" % (label, means.get(label, 0.0) - base.get(label, 0.0))
                for label in ("content", "newline", "whitespace", "punctuation",
                              "control", "all"))), flush=True)

    if digest_of(model) != backbone_digest:
        raise SystemExit("the frozen backbone changed")
    if not args.no_structure and digest_of(sidecar, skip="") != sidecar_digest:
        raise SystemExit("the frozen structural sidecar changed")

    report["elapsed_seconds"] = time.monotonic() - started
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2),
                                             encoding="utf-8")
    torch.save({"step": 0, "family": payload["family"], "layers": payload["layers"],
                "state_dict": {key: value.detach().cpu()
                               for key, value in handle.gates.state_dict().items()}},
               args.output / "gate.pt")
    remove_residual_gates(model)
    print("wrote %s in %.0f s" % (args.output, report["elapsed_seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
