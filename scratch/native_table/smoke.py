"""What does the reference-dense native table actually cost to train on this hardware?

The intended first regime, measured rather than estimated: real 2B student, backbone
frozen, table and PLE and rho trainable, bf16, gradient checkpointing, chunked
cross-entropy head, CE only -- no teacher, no cosine, no GR, no QSA.

The optimizer step is taken inside the measurement because AdamW allocates its moments
lazily on the first step, and those moments are the number that decides whether a dense
269M-element table is affordable at all. Sequence length climbs one rung at a time so an
OOM lands on a known rung instead of at the top.

    python scratch/native_table/smoke.py --model ../student-2b-hf --seq 1024 2048 4096
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

DEFAULT_MODEL = "D:/DeepThought/Projects/HybridModel/student-2b-hf"


def build(model_path, base, device, freeze_backbone=True):
    from transformers import AutoConfig

    from distillkit.chunked_ce import chunked_causal_lm_loss
    from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.sidecar_variant = "ple"
    config.sidecar_table_mode = "native"
    config.sidecar_ngram_vocab_size_base = base
    config.sidecar_layer_index = 1
    config.use_cache = False
    model = Qwen35SidecarForCausalLM.from_pretrained(
        model_path, config=config, dtype=torch.bfloat16,
        local_files_only=True).to(device)
    model.config.use_cache = False
    # The 248,320-wide head's fp32 upcast is the largest single activation in this model;
    # the chunked loss is what the real trainer uses and it belongs in the measurement.
    model.loss_function = chunked_causal_lm_loss
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    sidecar = model.model.layers[config.sidecar_layer_index].sidecar
    if freeze_backbone:
        model.requires_grad_(False)
        sidecar.requires_grad_(True)
    return model, sidecar, config


def measure(model, sidecar, hasher, tokens, batch, lr, device, steps=2):
    """One configuration, through a real optimizer step, reporting peak memory."""
    ids = torch.arange(1000, 1000 + tokens, dtype=torch.long) % 200000
    rows = hasher.row_indices(ids.unsqueeze(0).expand(batch, -1).contiguous())
    inputs = ids.unsqueeze(0).expand(batch, -1).contiguous().to(device)
    labels = inputs.clone()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    ngram_ids=rows.to(device), labels=labels)
        out.loss.backward()
        optimizer.step()
        losses.append(float(out.loss.detach()))
        del out

    def total(tensors):
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)

    state = sum(v.numel() * v.element_size()
                for entry in optimizer.state.values() for v in entry.values()
                if torch.is_tensor(v))
    table_state = sum(v.numel() * v.element_size()
                      for v in optimizer.state[sidecar.table.weight].values()
                      if torch.is_tensor(v))
    return {
        "tokens": tokens, "batch": batch, "loss": losses,
        "weights_bytes": total(list(model.parameters())),
        "table_weight_bytes": sidecar.table.weight.numel() * sidecar.table.weight.element_size(),
        "gradient_bytes": total([p.grad for p in trainable]),
        "optimizer_state_bytes": state,
        "table_optimizer_state_bytes": table_state,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base", type=int, default=131072)
    parser.add_argument("--seq", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--batch", type=int, nargs="+", default=[1])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from distillkit.native_ple import native_hash_config
    from distillkit.ngram_hash import NGramHasher

    torch.manual_seed(0)
    model, sidecar, config = build(args.model, args.base, args.device,
                                   freeze_backbone=not args.train_backbone)
    hasher = NGramHasher(native_hash_config(config))
    _, capacity = torch.cuda.mem_get_info(torch.device(args.device).index or 0)
    report = {
        "model": args.model, "device": args.device,
        "device_capacity_bytes": int(capacity),
        "backbone_frozen": not args.train_backbone,
        "geometry": {key: value for key, value in sidecar.geometry().items()
                     if key not in ("head_vocab_sizes", "head_offsets")},
        "parameters": {
            "model_total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "table": sidecar.table.weight.numel(),
            "ple": sum(p.numel() for p in sidecar.ple.parameters()),
        },
        "configurations": [],
    }

    for batch in args.batch:
        for tokens in args.seq:
            try:
                result = measure(model, sidecar, hasher, tokens, batch, args.lr, args.device)
                result["status"] = "ok"
            except torch.OutOfMemoryError as error:
                torch.cuda.empty_cache()
                result = {"tokens": tokens, "batch": batch, "status": "oom",
                          "error": str(error).split("\n")[0]}
            report["configurations"].append(result)
            print(json.dumps(result), flush=True)

    # Round trip once, on the model that has actually trained.
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM

        reloaded = Qwen35SidecarForCausalLM.from_pretrained(
            directory, dtype=torch.bfloat16).eval()
        restored = reloaded.model.layers[config.sidecar_layer_index].sidecar
        saved, back = model.state_dict(), reloaded.state_dict()
        report["reload"] = {
            "geometry_matches": restored.geometry() == sidecar.geometry(),
            "rho_matches": float(restored.rho) == float(sidecar.rho),
            "tensors_differing": [name for name in saved if name in back
                                  and not torch.equal(saved[name].cpu(), back[name].cpu())],
            "checkpoint_bytes": sum(path.stat().st_size
                                    for path in Path(directory).glob("*.safetensors")),
        }
    report["metrics"] = sidecar.gate_report()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("parameters", "reload", "metrics")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
