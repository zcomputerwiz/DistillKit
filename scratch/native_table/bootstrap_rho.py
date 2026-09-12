"""Open the admission gate with exactly one optimizer update, then stop.

The native block starts at ``rho = 0``, which makes the morphed model an exact copy of the
pretrained checkpoint -- and also means the table, the projections and the convolution
receive no gradient at all until ``rho`` itself has moved. Under the main regime (4096
tokens x accumulation 8) the first optimizer update lands after 32,768 tokens, so a short
experiment would spend a third of a 100K screen training one scalar.

So the gate is opened separately, with accumulation 1: one forward, one backward, one
update, roughly 4K tokens. The result is written as an ordinary checkpoint that the main
run starts from, and the token cost is reported rather than folded into the experiment's
own accounting.

Nothing is hand-initialised. ``rho`` starts at exactly zero and moves because the loss
asks it to, which is the only way the run can still claim to begin as an exact morph.

    python scratch/native_table/bootstrap_rho.py --output ../runs/native-ple-2b-bootstrap
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

DEFAULT_MODEL = "D:/DeepThought/Projects/HybridModel/student-2b-hf"
DEFAULT_DOCUMENTS = "D:/DeepThought/Projects/HybridModel/capture-data/heldout.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--documents", default=DEFAULT_DOCUMENTS)
    parser.add_argument("--output", required=True, type=Path,
                        help="where the bootstrapped checkpoint is written")
    parser.add_argument("--base", type=int, default=131072)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    from transformers import AutoConfig, AutoTokenizer

    from distillkit.chunked_ce import chunked_causal_lm_loss
    from distillkit.frozen_prefix import no_grad_prefix
    from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM
    from distillkit.native_ple import native_hash_config
    from distillkit.ngram_hash import NGramHasher

    torch.manual_seed(42)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.sidecar_variant = "ple"
    config.sidecar_table_mode = "native"
    config.sidecar_ngram_vocab_size_base = args.base
    config.sidecar_layer_index = 1
    config.use_cache = False

    model = Qwen35SidecarForCausalLM.from_pretrained(
        args.model, config=config, dtype=torch.bfloat16,
        local_files_only=True).to(args.device)
    model.config.use_cache = False
    model.loss_function = chunked_causal_lm_loss
    sidecar = model.model.layers[config.sidecar_layer_index].sidecar
    model.requires_grad_(False)
    sidecar.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    restore = no_grad_prefix(model, upto_layer=config.sidecar_layer_index)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    text = []
    with open(args.documents, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                text.append(json.loads(line)["text"])
            if len(text) > 8:
                break
    ids = tokenizer("\n\n".join(text))["input_ids"][:args.tokens]
    hasher = NGramHasher(native_hash_config(config))
    rows = hasher.row_indices(torch.tensor([ids], dtype=torch.long)).to(args.device)
    inputs = torch.tensor([ids], device=args.device)

    backbone = {name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
                if ".sidecar." not in name}
    tracked = {"rho": sidecar.rho, "table": sidecar.table.weight,
               "key_proj": sidecar.ple.key_proj.weight,
               "value_proj": sidecar.ple.value_proj.weight,
               "conv1d": sidecar.ple.conv1d.weight}

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    model.train()

    report = {"model": args.model, "tokens": len(ids), "lr": args.lr,
              "rho_before": float(sidecar.rho)}
    assert report["rho_before"] == 0.0, "the bootstrap must start from an exact morph"

    optimizer.zero_grad(set_to_none=True)
    out = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                ngram_ids=rows, labels=inputs.clone())
    out.loss.backward()
    report["loss"] = float(out.loss.detach())
    report["gradients_before_update"] = {
        name: (0.0 if parameter.grad is None else float(parameter.grad.detach().float().norm()))
        for name, parameter in tracked.items()}
    optimizer.step()

    report["rho_after"] = float(sidecar.rho)
    report["rho_moved"] = report["rho_after"] != 0.0
    report["rho_finite"] = bool(torch.isfinite(sidecar.rho).all())
    report["backbone_unchanged"] = all(
        torch.equal(backbone[name], parameter.detach())
        for name, parameter in model.named_parameters() if ".sidecar." not in name)

    # One more backward, no update: the question is whether the internals can now learn.
    optimizer.zero_grad(set_to_none=True)
    out = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                ngram_ids=rows, labels=inputs.clone())
    out.loss.backward()
    report["gradients_after_update"] = {
        name: (0.0 if parameter.grad is None else float(parameter.grad.detach().float().norm()))
        for name, parameter in tracked.items()}
    report["internals_now_learn"] = all(
        report["gradients_after_update"][name] > 0
        for name in ("table", "key_proj", "value_proj", "conv1d"))
    report["all_finite"] = all(
        bool(torch.isfinite(parameter.detach()).all()) for parameter in tracked.values())

    restore()
    model.zero_grad(set_to_none=True)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    report["checkpoint"] = str(args.output)
    report["bootstrap_tokens"] = len(ids)

    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not (report["rho_moved"] and report["backbone_unchanged"]
            and report["internals_now_learn"]):
        return 1
    print("\nbootstrap complete: %d tokens, rho %.3e -> %.6e"
          % (report["bootstrap_tokens"], 0.0, report["rho_after"]))
    print("the main run starts from %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
