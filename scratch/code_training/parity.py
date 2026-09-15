"""Check this harness applies G' and S exactly as the established scorers do.

The pre-training result is that both modules make held-out Python *worse* -- the sidecar
by 0.04 nats aggregate and 0.23 on newline, a class where it gained 0.47 on general text.
A sign inversion that large is either a real and important finding or a bug in how this
harness wires the modules up, and those two look identical in a table.

So the wiring is checked against the code that produced the published general-text
numbers. Three things are asserted, each of which would independently explain a false
negative:

*The sidecar's biased logits match* ``gate_after_structure.corrected`` token for token,
on real documents, which is the function every structural-sidecar result in this
programme was computed through.

*Forcing the gate to unit admission reproduces the stock model exactly.* If it does not,
something other than the gate differs between the arms and no delta means anything.

*The gate arm differs from stock only where the gate is installed.* A gate that changed
nothing would also produce a null result, for the opposite reason.

    CUDA_VISIBLE_DEVICES=0 python scratch/code_training/parity.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffn_memo"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffn_attenuation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "residual_gate"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "structural_sidecar"))

import numpy as np
import torch

from distillkit.experimental.residual_gate import (
    TrigramFamiliarity, install_residual_gates, remove_residual_gates)
from distillkit.experimental.structural_sidecar import structural_token_ids
from corpus import BASE, TOKENS, TokenStore, load_tokenizer
from evaluate_python import FAMILIARITY_CACHE, GATE, Sidecar


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path,
                        default=Path("scratch/code_training/manifests/parity.json"))
    args = parser.parse_args()

    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from distillkit.independent_eval import build_token_classes
    from evaluate import split_layout
    from factorize import class_ids
    from gate_after_structure import build_structural, corrected

    config = AutoConfig.from_pretrained(BASE, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.use_cache = False
    model = Qwen3_5ForCausalLM.from_pretrained(
        BASE, config=config, dtype=torch.bfloat16,
        local_files_only=True).to(args.device).eval()
    model.requires_grad_(False)

    tokenizer = load_tokenizer()
    classes = split_layout(build_token_classes(tokenizer, config.vocab_size), tokenizer,
                           config.vocab_size)
    structural = structural_token_ids(classes, device=args.device)
    whitespace = class_ids(classes, "whitespace", device=args.device)
    module, hasher, strength = build_structural(config, structural, whitespace, args.device)
    sidecar = Sidecar(module, hasher, structural, strength)

    store = TokenStore(TOKENS, "heldout")
    documents = []
    for index in range(len(store)):
        row = np.asarray(store.document(index)[:args.max_length], dtype=np.int64)
        if len(row) >= 64:
            documents.append(row)
        if len(documents) >= args.documents:
            break

    report = {"structural_tokens": int(structural.numel()),
              "whitespace_strength": float(strength),
              "sidecar_max_abs_difference": 0.0,
              "sidecar_changed_logits": 0,
              "identity_max_abs_difference": 0.0,
              "gate_changed_logits": 0}

    with torch.inference_mode():
        for row in documents:
            ids = torch.from_numpy(row).unsqueeze(0).to(args.device)
            mask = torch.ones_like(ids)

            # Reference: the function every published structural-sidecar number came from.
            reference = corrected(model, list(row), args.device, structural, module,
                                  hasher, strength)
            hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
            rows = sidecar.rows(ids)
            logits = model.lm_head(hidden[:, :len(row) - 1]).float()
            mine = sidecar.apply(logits, rows, 0, len(row) - 1)[0]
            report["sidecar_max_abs_difference"] = max(
                report["sidecar_max_abs_difference"],
                float((mine - reference).abs().max()))

            plain = model.lm_head(hidden[:, :len(row) - 1]).float()[0]
            report["sidecar_changed_logits"] += int((mine != plain).sum())

        # The same-checkpoint ablation: admission forced to 1 must be the stock model.
        payload = torch.load(GATE, map_location="cpu", weights_only=False)
        familiarity = TrigramFamiliarity(FAMILIARITY_CACHE, config.vocab_size,
                                         device=args.device)
        handle = install_residual_gates(model, payload["layers"],
                                        family=payload["family"],
                                        familiarity=familiarity)
        model.residual_gates.load_state_dict(payload["state_dict"], strict=True)
        model.residual_gates.requires_grad_(False)
        for row in documents:
            ids = torch.from_numpy(row).unsqueeze(0).to(args.device)
            mask = torch.ones_like(ids)
            handle.force_identity = False
            handle.set_context(ids)
            learned = model(input_ids=ids, attention_mask=mask).logits[0, :-1].float()
            handle.force_identity = True
            handle.set_context(ids)
            identity = model(input_ids=ids, attention_mask=mask).logits[0, :-1].float()
            handle.force_identity = False
            remove_residual_gates(model)
            stock = model(input_ids=ids, attention_mask=mask).logits[0, :-1].float()
            handle = install_residual_gates(model, payload["layers"],
                                            family=payload["family"],
                                            familiarity=familiarity)
            model.residual_gates.load_state_dict(payload["state_dict"], strict=True)
            report["identity_max_abs_difference"] = max(
                report["identity_max_abs_difference"],
                float((identity - stock).abs().max()))
            report["gate_changed_logits"] += int((learned != stock).sum())
        remove_residual_gates(model)

    report["sidecar_matches_reference"] = report["sidecar_max_abs_difference"] < 1e-3
    report["identity_recovers_stock"] = report["identity_max_abs_difference"] < 1e-3
    report["gate_is_active"] = report["gate_changed_logits"] > 0
    report["sidecar_is_active"] = report["sidecar_changed_logits"] > 0

    print("sidecar vs gate_after_structure.corrected: max |diff| %.3e  -> %s"
          % (report["sidecar_max_abs_difference"],
             "MATCH" if report["sidecar_matches_reference"] else "MISMATCH"))
    print("sidecar actually changed %d logits" % report["sidecar_changed_logits"])
    print("gate forced to identity vs stock:          max |diff| %.3e  -> %s"
          % (report["identity_max_abs_difference"],
             "MATCH" if report["identity_recovers_stock"] else "MISMATCH"))
    print("gate actually changed %d logits" % report["gate_changed_logits"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    ok = (report["sidecar_matches_reference"] and report["identity_recovers_stock"]
          and report["gate_is_active"] and report["sidecar_is_active"])
    print("\n%s" % ("PARITY OK -- the arms differ only by the modules"
                    if ok else "PARITY FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
