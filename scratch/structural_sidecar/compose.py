"""Do the two post-hoc corrections add up, or are they doing the same job twice?

Two mechanisms have now survived on a frozen backbone, and they were fitted independently
and act in different places:

    the scalar residual gate    scales one FFN's contribution by context familiarity
    the structural sidecar      biases structural logits by hashed local context

Nothing guarantees those compose. Both were fitted against the same stock backbone and
both key off local context, so the honest possibilities are additive, redundant, or
interfering, and only the four-way measurement distinguishes them. Neither module trains
here; both are loaded and frozen, and the backbone was never trainable in either fitting.

    CUDA_VISIBLE_DEVICES=0 python scratch/structural_sidecar/compose.py \
        --backbone B42 --output scratch/structural_sidecar/compose-B42.json
"""

from __future__ import annotations

import argparse
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
    TrigramFamiliarity, family_features, install_residual_gates,
    remove_residual_gates)
from distillkit.experimental.structural_sidecar import (
    FactorizedSidecar, StructuralSidecar, apply_structural_bias, structural_token_ids)
from evaluate import DEFAULT_CACHE, split_layout
from fit import BACKBONES, BASE
from repeatability import DEFAULT_BUNDLE

GATE = Path("scratch/residual_gate/gates/familiarity-lr3e3-step284.pt")


def paired(left, right):
    values = [a - b for a, b in zip(left, right)]
    mean = statistics.fmean(values)
    error = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
    return {"mean": mean, "t": mean / error if error else 0.0,
            "better": sum(1 for value in values if value < 0), "n": len(values)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="B42", choices=sorted(BACKBONES))
    parser.add_argument("--sidecar", type=Path,
                        default=Path("scratch/structural_sidecar/fixed-B42/sidecar.pt"))
    parser.add_argument("--gate", type=Path, default=GATE)
    parser.add_argument("--whitespace", type=Path, default=None,
                        help="a fitted whitespace bias; factorizes the sidecar first")
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
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

    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    classes = split_layout(build_token_classes(tokenizer, config.vocab_size), tokenizer,
                           config.vocab_size)
    structural = structural_token_ids(classes, device=args.device)

    saved = torch.load(args.sidecar, map_location="cpu", weights_only=False)
    sidecar = StructuralSidecar(rows=saved["rows"], code_dim=saved["code_dim"],
                                structural=int(structural.numel()), mode=saved["mode"],
                                heads=saved["heads"], hidden=saved["hidden"],
                                seed=saved["seed"]).to(args.device).to(torch.float32)
    sidecar.load_state_dict(saved["state_dict"])
    sidecar.requires_grad_(False)
    strength = saved["strength"]
    whitespace_strength = 0.0
    if args.whitespace is not None:
        payload = torch.load(args.whitespace, map_location="cpu", weights_only=False)
        ids = [index for index, label in enumerate(classes) if label == "whitespace"]
        sidecar = FactorizedSidecar(sidecar, structural.cpu(),
                                    torch.tensor(ids, dtype=torch.long)).to(args.device)
        sidecar.white.load_state_dict(payload["state_dict"])
        sidecar.requires_grad_(False)
        whitespace_strength = payload["whitespace_strength"] / strength
    hasher = NGramHasher(NGramHashConfig(
        vocab_size=config.vocab_size, ngram_size=3, heads_per_ngram=1,
        ngram_vocab_size_base=saved["rows"] // 2 - 64, seed=1234,
        eos_token_id=tokenizer.eos_token_id or config.vocab_size - 1))
    if hasher.padded_vocab_size != saved["rows"]:
        raise SystemExit("hash geometry %d does not match the saved sidecar's %d"
                         % (hasher.padded_vocab_size, saved["rows"]))

    payload = torch.load(args.gate, map_location="cpu", weights_only=False)
    report = {"backbone": args.backbone, "checkpoint": str(checkpoint),
              "sidecar": str(args.sidecar), "sidecar_strength": strength,
              "gate": str(args.gate), "splits": {}}

    for split in ("screen", "confirmation"):
        documents = [record["ids"] for record in load_records(args.bundle, split, 0)]
        series = {}
        for gate_on in (False, True):
            handle = None
            if gate_on:
                source = (TrigramFamiliarity(args.cache, config.vocab_size)
                          if "log_count" in family_features(payload["family"]) else None)
                handle = install_residual_gates(model, payload["layers"],
                                                family=payload["family"],
                                                familiarity=source)
                model.residual_gates.load_state_dict(payload["state_dict"], strict=True)
            try:
                for sidecar_on in (False, True):
                    name = "%s%s" % ("gate" if gate_on else "",
                                     "+sidecar" if sidecar_on else "")
                    name = name.strip("+") or "stock"
                    per_document = {}
                    with torch.no_grad():
                        for ids in documents:
                            tokens = torch.tensor([ids], device=args.device)
                            logits = model(
                                input_ids=tokens,
                                attention_mask=torch.ones_like(tokens)
                            ).logits[0, :-1].float()
                            if sidecar_on:
                                rows = hasher.row_indices(tokens)
                                bias = (sidecar(rows, whitespace_strength)
                                        if args.whitespace else sidecar(rows))[0, :-1]
                                logits = apply_structural_bias(
                                    logits.unsqueeze(0), bias.unsqueeze(0), structural,
                                    strength)[0]
                            targets = torch.tensor(ids[1:], device=args.device)
                            nll = F.cross_entropy(logits, targets,
                                                  reduction="none").cpu()
                            sums = {}
                            for index, token in enumerate(ids[1:]):
                                entry = sums.setdefault(classes[token], [0.0, 0])
                                entry[0] += float(nll[index])
                                entry[1] += 1
                            sums["all"] = [float(nll.sum()), len(ids) - 1]
                            for label, (value, count) in sums.items():
                                per_document.setdefault(label, []).append(value / count)
                    series[name] = per_document
                    print("%-12s %-14s content %.6f" % (
                        split, name, statistics.fmean(per_document["content"])),
                        flush=True)
            finally:
                if gate_on:
                    remove_residual_gates(model)

        entry = {"nll": {name: {label: statistics.fmean(values)
                                for label, values in values_by_label.items()}
                         for name, values_by_label in series.items()},
                 "differences": {}}
        for name in ("gate", "sidecar", "gate+sidecar"):
            entry["differences"]["%s - stock" % name] = {
                label: paired(series[name][label], series["stock"][label])
                for label in series["stock"]}
        # Additive if the pair moves by the sum of the singles; this is the residual.
        entry["interaction"] = {
            label: (entry["differences"]["gate+sidecar - stock"][label]["mean"]
                    - entry["differences"]["gate - stock"][label]["mean"]
                    - entry["differences"]["sidecar - stock"][label]["mean"])
            for label in series["stock"]}
        report["splits"][split] = entry

    report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for split, entry in report["splits"].items():
        print("\n%s" % split)
        for label in ("content", "newline", "whitespace", "punctuation", "control",
                      "all"):
            print("  %-12s gate %+.6f  sidecar %+.6f  both %+.6f  interaction %+.6f"
                  % (label,
                     entry["differences"]["gate - stock"][label]["mean"],
                     entry["differences"]["sidecar - stock"][label]["mean"],
                     entry["differences"]["gate+sidecar - stock"][label]["mean"],
                     entry["interaction"][label]))
    print("\nwrote %s in %.0f s" % (args.output, report["elapsed_seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
