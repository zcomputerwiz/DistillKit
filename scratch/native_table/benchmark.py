"""How fast is the optimized path, and which microbatch is worth using?

Correctness is settled elsewhere (`tests/test_frozen_prefix.py`); this only asks what the
configurations cost. Every measured configuration is the real regime: converted 2B
student, native table, frozen backbone with the prefix out of autograd, CE only, bf16,
chunked head, a real AdamW step each iteration.

Optimizer state is allocated before timing starts, because AdamW builds its moments on the
first step and that step would otherwise carry a gigabyte of allocation into the average.
Kernels are warmed for the same reason -- the linear-attention path autotunes on first use.

    python scratch/native_table/benchmark.py --steps 12
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from vram_guard import Spilled, baseline, cap, spill_check

DEFAULT_MODEL = "D:/DeepThought/Projects/HybridModel/student-2b-hf"


def build(model_path, base, device, checkpointing):
    from transformers import AutoConfig

    from distillkit.chunked_ce import chunked_causal_lm_loss
    from distillkit.frozen_prefix import no_grad_prefix
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
    model.loss_function = chunked_causal_lm_loss
    sidecar = model.model.layers[config.sidecar_layer_index].sidecar
    model.requires_grad_(False)
    sidecar.requires_grad_(True)
    if checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()
    restore = no_grad_prefix(model, upto_layer=config.sidecar_layer_index)
    model.train()
    return model, sidecar, config, restore


def run(model, sidecar, hasher, tokens, batch, lr, device, steps, warmup=3,
        budget=0.85):
    ids = (torch.arange(1000, 1000 + tokens, dtype=torch.long) % 200000)
    inputs = ids.unsqueeze(0).expand(batch, -1).contiguous().to(device)
    rows = hasher.row_indices(ids.unsqueeze(0).expand(batch, -1).contiguous()).to(device)
    labels = inputs.clone()
    mask = torch.ones_like(inputs)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)

    def iteration():
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=inputs, attention_mask=mask, ngram_ids=rows, labels=labels)
        out.loss.backward()
        optimizer.step()
        return float(out.loss.detach())

    capacity = torch.cuda.mem_get_info(torch.device(device).index or 0)[1]
    baseline_rss = baseline()

    torch.cuda.reset_peak_memory_stats()
    losses = []
    for step in range(warmup):                        # also allocates optimizer state
        losses.append(iteration())
        torch.cuda.synchronize()
        spill_check(budget, capacity, baseline_rss, "warmup step %d" % step)

    torch.cuda.reset_peak_memory_stats()
    durations = []
    for step in range(steps):
        start = time.perf_counter()
        losses.append(iteration())
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
        spill_check(budget, capacity, baseline_rss, "step %d" % step)

    per_step = statistics.median(durations)
    return {
        "tokens": tokens, "batch": batch, "steps": steps,
        "mean_step_seconds": statistics.fmean(durations),
        "median_step_seconds": per_step,
        "min_step_seconds": min(durations),
        "tokens_per_second": tokens * batch / per_step,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "loss_first": losses[0], "loss_last": losses[-1],
        "all_finite": all(value == value for value in losses),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base", type=int, default=131072)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpointing", choices=["on", "off", "both"], default="both",
                        help="which mode to measure; 'off' at batch > 1 spills 39 GiB "
                             "onto a 24 GiB card and is measured once, not repeatedly")
    parser.add_argument("--budget", type=float, default=0.85,
                        help="fraction of the card a configuration may reserve "
                             "after warmup before it is rejected as spilling")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from distillkit.native_ple import native_hash_config
    from distillkit.ngram_hash import NGramHasher

    # CUDA_VISIBLE_DEVICES=0 is the whole isolation story, so record what the process
    # can actually see. If the second card shows dedicated memory in use while this runs,
    # it belongs to some other process, not to this one.
    visible = torch.cuda.device_count()
    if visible != 1:
        raise SystemExit("benchmark expects one visible GPU, found %d; run with "
                         "CUDA_VISIBLE_DEVICES=0" % visible)
    index, capacity = cap(args.device, args.budget)
    report = {"model": args.model, "tokens": args.tokens,
              "device_capacity_bytes": int(capacity),
              "device_name": torch.cuda.get_device_name(index),
              "visible_devices": visible, "memory_fraction": args.budget, "results": []}
    print("device: %s  visible=%d  cap=%.1f GiB  fraction=%.2f"
          % (report["device_name"], visible, capacity / 2**30, args.budget), flush=True)

    modes = {"on": (True,), "off": (False,), "both": (False, True)}[args.checkpointing]
    for checkpointing in modes:
        torch.manual_seed(0)
        model, sidecar, config, restore = build(args.model, args.base, args.device,
                                                checkpointing)
        hasher = NGramHasher(native_hash_config(config))
        report["trainable_parameters"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        for batch in args.batch:
            label = "checkpointing_%s_batch_%d" % ("on" if checkpointing else "off", batch)
            try:
                result = run(model, sidecar, hasher, args.tokens, batch, args.lr,
                             args.device, args.steps, budget=args.budget)
                result["status"] = "ok"
            except (torch.OutOfMemoryError, Spilled) as error:
                torch.cuda.empty_cache()
                result = {"tokens": args.tokens, "batch": batch,
                          "status": "spilled" if isinstance(error, Spilled) else "oom",
                          "error": str(error).split("\n")[0]}
            result["checkpointing"] = checkpointing
            result["label"] = label
            report["results"].append(result)
            print(json.dumps(result), flush=True)
        restore()
        del model, sidecar
        torch.cuda.empty_cache()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n%-28s %10s %12s %10s %10s" % ("configuration", "tok/s", "step (s)",
                                           "alloc GiB", "resv GiB"))
    for result in report["results"]:
        if result["status"] != "ok":
            print("%-28s %10s  %s" % (result["label"], result["status"].upper(),
                                      result.get("error", "")))
            continue
        print("%-28s %10.1f %12.3f %10.2f %10.2f"
              % (result["label"], result["tokens_per_second"],
                 result["median_step_seconds"],
                 result["peak_allocated_bytes"] / 2**30,
                 result["peak_reserved_bytes"] / 2**30))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
