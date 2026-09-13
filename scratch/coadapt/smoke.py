"""Can a trainable 2B backbone plus the native table fit on one 3090, and do the freezes hold?

Two arms start from the same checkpoint and must be measured before either is launched:

    A   backbone + table + PLE block + rho all trainable, PLE on, correct addressing
    B   backbone trainable, PLE bypassed in forward and frozen in place

The frozen stage trained 277M parameters and reserved 13.7 GiB. This stage trains all
2.15B, which adds bf16 gradients for every one of them plus 8-bit AdamW state, so the
question is not rhetorical. Anything that reserves more than the budget is rejected here
rather than discovered by the desktop grinding to a halt: Windows does not raise on an
oversized allocation, it pages to host RAM and keeps going about seventy times slower.

Three things are checked, because a run that fits but trains the wrong tensors is worse
than one that does not fit:

    memory and throughput   through a real 8-bit AdamW step, after the lazy state exists
    gradient contracts      every tensor that must learn has a finite nonzero gradient
    freeze contracts        arm B's table and PLE block are bitwise unchanged by a step

    python scratch/coadapt/smoke.py --arm A --output scratch/coadapt/smoke-A.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native_table"))

import torch

from vram_guard import Spilled, baseline, cap, spill_check

DEFAULT_CHECKPOINT = "D:/DeepThought/Projects/HybridModel/runs/native-ple-2b-ce-1m/checkpoint-675"

# Gradients are checked at the edges and the middle of the stack, not everywhere: a
# backbone that learns at layer 0 and layer 23 is not silently detached in between.
PROBE_LAYERS = (0, 12, 23)


def build(checkpoint, device, arm):
    from transformers import AutoConfig

    from distillkit.chunked_ce import chunked_causal_lm_loss
    from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM
    from distillkit.optimizers import freeze_sidecar_parameters

    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.use_cache = False
    model = Qwen35SidecarForCausalLM.from_pretrained(
        checkpoint, config=config, dtype=torch.bfloat16,
        local_files_only=True).to(device)
    model.config.use_cache = False
    model.loss_function = chunked_causal_lm_loss
    model.requires_grad_(True)                    # the backbone trains in both arms
    frozen = ()
    if arm == "B":
        frozen = freeze_sidecar_parameters(model)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model, config, frozen


def tracked(model, config):
    sidecar = model.model.layers[config.sidecar_layer_index].sidecar
    names = {"rho": sidecar.rho, "table": sidecar.table.weight,
             "key_proj": sidecar.ple.key_proj.weight,
             "value_proj": sidecar.ple.value_proj.weight,
             "conv1d": sidecar.ple.conv1d.weight}
    for layer in PROBE_LAYERS:
        # mlp.down_proj, not an attention projection: the stack mixes linear_attention
        # and full_attention layers, which do not share a module name.
        names["layer%d.mlp" % layer] = model.model.layers[layer].mlp.down_proj.weight
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["A", "B"], required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--sidecar-lr", type=float, default=1e-4)
    parser.add_argument("--budget", type=float, default=0.90,
                        help="fraction of the card this process may take; enforced")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    visible = torch.cuda.device_count()
    if visible != 1:
        raise SystemExit("run one arm per GPU with CUDA_VISIBLE_DEVICES; saw %d devices"
                         % visible)
    index, capacity = cap(args.device, args.budget)

    from bitsandbytes.optim import AdamW8bit

    from distillkit.native_ple import native_hash_config
    from distillkit.ngram_hash import NGramHasher

    torch.manual_seed(0)
    model, config, frozen = build(args.checkpoint, args.device, args.arm)
    hasher = NGramHasher(native_hash_config(config))

    ids = torch.arange(1000, 1000 + args.tokens, dtype=torch.long) % 200000
    inputs = ids.unsqueeze(0).expand(args.batch, -1).contiguous().to(args.device)
    rows = hasher.row_indices(
        ids.unsqueeze(0).expand(args.batch, -1).contiguous()).to(args.device)
    mask = torch.ones_like(inputs)

    watched = tracked(model, config)
    before = {name: parameter.detach().clone() for name, parameter in watched.items()}

    # Two groups, which is what keeps a pretrained backbone from being shoved around at
    # the rate a randomly initialised table needs.
    sidecar_prefix = "model.layers.%d.sidecar." % config.sidecar_layer_index
    backbone_params, sidecar_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (sidecar_params if name.startswith(sidecar_prefix) else backbone_params
         ).append(parameter)
    groups = [{"params": backbone_params, "lr": args.lr}]
    if sidecar_params:
        groups.append({"params": sidecar_params, "lr": args.sidecar_lr})
    optimizer = AdamW8bit(groups, lr=args.lr, weight_decay=0.0)

    def iteration():
        optimizer.zero_grad(set_to_none=True)
        kwargs = {"input_ids": inputs, "attention_mask": mask, "labels": inputs.clone()}
        if args.arm == "A":
            kwargs["ngram_ids"] = rows
        else:
            kwargs["sidecar_enabled"] = False
        out = model(**kwargs)
        out.loss.backward()
        optimizer.step()
        return float(out.loss.detach())

    report = {"arm": args.arm, "checkpoint": args.checkpoint,
              "device_name": torch.cuda.get_device_name(index),
              "visible_devices": visible, "tokens": args.tokens, "batch": args.batch,
              "backbone_lr": args.lr, "sidecar_lr": args.sidecar_lr if sidecar_params else None,
              "optimizer": "bitsandbytes AdamW8bit",
              "trainable_parameters": sum(p.numel() for p in model.parameters()
                                          if p.requires_grad),
              "frozen_sidecar_tensors": len(frozen)}

    baseline_rss = baseline()
    torch.cuda.reset_peak_memory_stats()
    losses = []
    try:
        for step in range(args.warmup):          # also allocates the 8-bit state
            losses.append(iteration())
            torch.cuda.synchronize()
            spill_check(args.budget, capacity, baseline_rss, "warmup step %d" % step)
    except (torch.OutOfMemoryError, Spilled) as error:
        report["status"] = "spilled" if isinstance(error, Spilled) else "oom"
        report["error"] = str(error).split("\n")[0]
        print(json.dumps(report, indent=2))
        if args.output:
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 1

    # The gradients from the last warmup step are still attached; read them before the
    # timing loop zeroes them.
    report["gradients"] = {
        name: (None if parameter.grad is None
               else float(parameter.grad.detach().float().norm()))
        for name, parameter in watched.items()}

    torch.cuda.reset_peak_memory_stats()
    durations = []
    for step in range(args.steps):
        start = time.perf_counter()
        losses.append(iteration())
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
        spill_check(args.budget, capacity, baseline_rss, "step %d" % step)

    per_step = statistics.median(durations)
    report.update({
        "status": "ok",
        "median_step_seconds": per_step,
        "tokens_per_second": args.tokens * args.batch / per_step,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "capacity_gib": capacity / 2**30,
        "loss_first": losses[0], "loss_last": losses[-1],
        "all_finite": all(value == value for value in losses),
    })

    # Contracts. Arm A: everything named must learn. Arm B: the backbone must learn and
    # the sidecar must not have moved by so much as a bit.
    sidecar_names = ("rho", "table", "key_proj", "value_proj", "conv1d")
    backbone_names = tuple("layer%d.mlp" % layer for layer in PROBE_LAYERS)
    grads = report["gradients"]
    report["backbone_learns"] = all(
        grads[name] is not None and grads[name] > 0 for name in backbone_names)
    if args.arm == "A":
        report["sidecar_learns"] = all(
            grads[name] is not None and grads[name] > 0 for name in sidecar_names)
        report["contract_ok"] = report["backbone_learns"] and report["sidecar_learns"]
    else:
        report["sidecar_has_no_gradients"] = all(
            grads[name] is None for name in sidecar_names)
        report["sidecar_bitwise_unchanged"] = all(
            torch.equal(before[name], watched[name].detach()) for name in sidecar_names)
        report["contract_ok"] = (report["backbone_learns"]
                                 and report["sidecar_has_no_gradients"]
                                 and report["sidecar_bitwise_unchanged"])
    report["backbone_moved"] = any(
        not torch.equal(before[name], watched[name].detach()) for name in backbone_names)

    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["contract_ok"] and report["backbone_moved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
