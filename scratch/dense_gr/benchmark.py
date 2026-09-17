"""Measured training throughput for the dense-GR reference substrate.

The sizing discussion so far has run on one extrapolated number -- roughly 2,010 tok/s
for the 2B student, scaled by parameter count and hedged for the four residual streams
that measurement never included. That is not good enough to choose a model size with,
because the streams change memory traffic rather than FLOPs and the 248,320-entry head
dominates activations at any width.

This runs real optimizer steps: bf16 weights, 8-bit AdamW, gradient checkpointing,
chunked cross-entropy, random inputs, one card. It reports tokens per second, step time
and peak memory, and converts those into the wall-clock the copy experiment would
actually cost at 1B and 3B tokens per arm.

    CUDA_VISIBLE_DEVICES=0 python scratch/dense_gr/benchmark.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from distillkit.core.chunked_ce import maybe_install_chunked_loss
from distillkit.models import Qwen35WidenedForCausalLM

#: (label, hidden, layers, vocab). Two ~0.8B shapes at the same budget but different
#: depth/width splits, plus the small candidates for the first experiment.
CONFIGURATIONS = [
    ("d1536-L12", 1536, 12, 248_320),
    ("d1280-L18", 1280, 18, 248_320),
    ("d768-L12-v32k", 768, 12, 32_768),
    ("d512-L8-v16k", 512, 8, 16_384),
]


def build(hidden, layers, vocab, head_dim=64, branches=4, interval=2):
    heads = max(2, hidden // head_dim)
    pairs = int(head_dim * 0.25) // 2
    section = [pairs - 2 * (pairs // 3), pairs // 3, pairs // 3]
    config = Qwen3_5TextConfig(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=3 * hidden,
        num_hidden_layers=layers, num_attention_heads=heads,
        num_key_value_heads=max(1, heads // 4), head_dim=head_dim,
        linear_key_head_dim=head_dim, linear_value_head_dim=head_dim,
        linear_num_key_heads=max(1, hidden // head_dim // 2),
        linear_num_value_heads=max(1, hidden // head_dim // 2),
        linear_conv_kernel_dim=4, full_attention_interval=interval,
        tie_word_embeddings=True, max_position_embeddings=8192, pad_token_id=0,
        eos_token_id=2, use_cache=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0,
                         "partial_rotary_factor": 0.25, "mrope_interleaved": True,
                         "mrope_section": section},
    )
    config.residual_stream_enabled = True
    config.residual_stream_routing = "flash_next"
    config.residual_stream_num_branches = branches
    config.residual_stream_lowrank = max(8, hidden // 8)
    config.residual_stream_sidecar = False
    config.residual_stream_blend = 0.0
    return config


def measure(model, config, batch, length, steps, warmup, device="cuda"):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)

    tokens = torch.randint(0, config.vocab_size, (batch, length), device=device)
    attention = torch.ones_like(tokens)

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=tokens, attention_mask=attention, labels=tokens,
                     use_cache=False).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return float(loss)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    started = time.perf_counter()
    last = None
    for _ in range(steps):
        last = step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    reserved = torch.cuda.max_memory_reserved()
    budget = torch.cuda.get_device_properties(0).total_memory
    result = {
        "batch": batch, "length": length, "steps": steps,
        "seconds_per_step": elapsed / steps,
        "tokens_per_second": batch * length * steps / elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
        "peak_reserved_gib": reserved / 2 ** 30,
        "reserved_fraction_of_vram": reserved / budget,
        "loss": last,
    }
    del optimizer, tokens, attention
    for parameter in model.parameters():
        parameter.grad = None
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--vram-fraction", type=float, default=0.90,
                        help="cap PyTorch's share so over-budget raises instead of "
                             "spilling to shared system memory")
    parser.add_argument("--output", type=Path,
                        default=Path("scratch/dense_gr/benchmark.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    if torch.cuda.device_count() != 1:
        raise SystemExit("mask to one card with CUDA_VISIBLE_DEVICES")

    # The whole reason this script exists twice. On Windows WDDM an over-budget run does
    # not raise: the driver pages CUDA allocations to shared system memory and services
    # them over PCIe, which still reports 100% utilisation while running several times
    # slower. The first pass of this benchmark reserved 35.27 GiB on a 24 GiB card at
    # batch 16 and reported 1,714 tok/s against 4,491 at batch 4 -- a measurement of PCIe
    # bandwidth wearing the costume of a throughput number. Capping the allocator turns
    # that silent degradation into an OutOfMemoryError the batch sweep already handles.
    # `max_vram_fraction` in distillkit/configuration.py does the same thing for runs.
    torch.cuda.set_per_process_memory_fraction(args.vram_fraction, 0)

    report = {"device": torch.cuda.get_device_name(0), "length": args.length,
              "dtype": "bfloat16", "optimizer": "bnb.AdamW8bit",
              "gradient_checkpointing": not args.no_checkpointing,
              "chunked_cross_entropy": True,
              "vram_fraction": args.vram_fraction,
              "total_vram_gib": torch.cuda.get_device_properties(0).total_memory / 2 ** 30,
              "configurations": {}}

    for label, hidden, layers, vocab in CONFIGURATIONS:
        if args.only and label not in args.only:
            continue
        config = build(hidden, layers, vocab)
        with torch.device("meta"):
            counted = Qwen35WidenedForCausalLM(config)
        total = sum(p.numel() for p in counted.parameters())
        embed = counted.model.embed_tokens.weight.numel()
        del counted
        entry = {"hidden": hidden, "layers": layers, "vocab": vocab,
                 "parameters": total, "embedding_parameters": embed,
                 "core_parameters": total - embed, "runs": []}
        print("\n=== %s  %d layers x %d, vocab %d, %.1fM params (%.0f%% core) ==="
              % (label, layers, hidden, vocab, total / 1e6,
                 100 * (total - embed) / total), flush=True)
        # Built once and reused across the batch sweep. The previous pass rebuilt a
        # 0.8B model for every batch point, which cost more wall clock than the
        # measurements did and burned a 3.2 GB fp32 host copy each time.
        torch.manual_seed(0)
        model = Qwen35WidenedForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
        if not args.no_checkpointing:
            model.gradient_checkpointing_enable()
        model.train()
        maybe_install_chunked_loss(model, need_model_loss=True, enabled=True)
        for batch in args.batches:
            try:
                row = measure(model, config, batch, args.length, args.steps, args.warmup)
            except torch.OutOfMemoryError:
                print("  batch %-3d OOM (capped at %.0f%% of VRAM)"
                      % (batch, 100 * args.vram_fraction), flush=True)
                for parameter in model.parameters():
                    parameter.grad = None
                torch.cuda.empty_cache()
                entry["runs"].append({"batch": batch, "oom": True})
                break
            hours = lambda budget: budget / row["tokens_per_second"] / 3600
            row["hours_per_arm_1b"] = hours(1e9)
            row["hours_per_arm_3b"] = hours(3e9)
            entry["runs"].append(row)
            print("  batch %-3d %8.0f tok/s  %6.3f s/step  peak %5.2f GiB alloc / "
                  "%5.2f GiB reserved (%2.0f%% VRAM)  1B: %5.1f h  3B: %5.1f h"
                  % (batch, row["tokens_per_second"], row["seconds_per_step"],
                     row["peak_allocated_gib"], row["peak_reserved_gib"],
                     100 * row["reserved_fraction_of_vram"],
                     row["hours_per_arm_1b"], row["hours_per_arm_3b"]), flush=True)
        best = max((r for r in entry["runs"] if not r.get("oom")),
                   key=lambda r: r["tokens_per_second"], default=None)
        entry["best"] = best
        del model
        torch.cuda.empty_cache()
        report["configurations"][label] = entry
        # Written after every configuration, so a sweep that dies on the last one still
        # leaves the earlier measurements on disk.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nwrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
