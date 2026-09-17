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
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from distillkit.core.chunked_ce import maybe_install_chunked_loss
from distillkit.models import Qwen35WidenedForCausalLM
from distillkit.models.qwen35.gqa_dispatch import (
    fused_kernel_supports_gqa, install_expanded_gqa_attention)
from distillkit.models.qwen35.linear_attention_dispatch import (
    fused_linear_attention_available, install_device_aware_linear_attention)

def _shim_triton_metadata():
    """Let CCE find Triton's version on Windows.

    `cut_cross_entropy.utils.is_triton_3_2` reads
    `importlib.metadata.version("triton")` to decide which kernel path to take, but the
    Windows distribution is packaged as `triton-windows`, so that lookup raises
    PackageNotFoundError at call time rather than import time. Resolving the alias keeps
    CCE's own version comparison intact instead of hard-coding the answer.
    """
    import importlib.metadata as metadata

    original = metadata.version

    def version(name):
        try:
            return original(name)
        except metadata.PackageNotFoundError:
            if name == "triton":
                return original("triton-windows")
            raise

    metadata.version = version


_shim_triton_metadata()

try:
    from cut_cross_entropy import linear_cross_entropy
except ImportError:  # optional; --loss cce is simply unavailable
    linear_cross_entropy = None

#: (label, hidden, layers, vocab). Two ~0.8B shapes at the same budget but different
#: depth/width splits, plus the small candidates for the first experiment.
CONFIGURATIONS = [
    ("d1536-L12", 1536, 12, 248_320),
    ("d1280-L18", 1280, 18, 248_320),
    ("d768-L12-v32k", 768, 12, 32_768),
    ("d512-L8-v16k", 512, 8, 16_384),
]


def build(hidden, layers, vocab, head_dim=64, branches=4, interval=2,
          attn_implementation="sdpa"):
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
    # Only the full-attention layers consult this; the linear-attention layers go
    # through the gated delta rule and the conv, which fla and causal_conv1d own.
    config._attn_implementation = attn_implementation
    return config


def apply_liger(model, config):
    """Swap the stock SwiGLU and RMSNorm for Liger's fused kernels, where they fit.

    Two of the three obvious candidates fit without rework and one does not.

    *SwiGLU* fits cleanly: `down_proj(act(gate(x)) * up(x))` becomes one fused kernel for
    the activation and product, and only the forward is replaced, so no parameter moves
    and no checkpoint changes shape.

    *RMSNorm* fits because `LigerRMSNorm(offset=1.0, casting_mode="gemma")` is exactly
    Qwen3.5's convention -- it stores a deviation, applies `1 + weight`, and does the
    weight multiply in fp32 before casting back. Note this swaps the module *class*, so a
    model that has been Liger-patched will no longer satisfy `recipient_initialize`'s
    `Qwen3_5RMSNorm` type check. That refusal is deliberate and this is why it exists.

    *RoPE* does not fit. Qwen3.5 uses interleaved mRoPE with a partial rotary factor and
    Liger's kernel is the standard formulation; matching them is rework, not a swap.

    The branch norms inside the GR route are left alone on purpose. They are a custom
    autograd Function written to avoid holding fp32 copies until backward, the whole
    route is 4.7% of a step, and trading that memory design for a fused kernel is a bad
    exchange at that share.
    """
    import types

    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    from liger_kernel.transformers import LigerRMSNorm
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP, Qwen3_5RMSNorm

    def fused_mlp_forward(self, x):
        return self.down_proj(LigerSiLUMulFunction.apply(self.gate_proj(x),
                                                         self.up_proj(x)))

    swapped = {"swiglu": 0, "rmsnorm": 0}
    for module in model.modules():
        if isinstance(module, Qwen3_5MLP):
            module.forward = types.MethodType(fused_mlp_forward, module)
            swapped["swiglu"] += 1
    for name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, Qwen3_5RMSNorm):
                continue
            replacement = LigerRMSNorm(
                child.weight.shape[0], eps=float(child.eps), offset=1.0,
                casting_mode="gemma", init_fn="zeros").to(
                    device=child.weight.device, dtype=child.weight.dtype)
            replacement.weight.data.copy_(child.weight.data)
            setattr(parent, child_name, replacement)
            swapped["rmsnorm"] += 1
    return swapped


def _module_version(name):
    import importlib
    try:
        return getattr(importlib.import_module(name), "__version__", "present")
    except Exception:
        return None


def shared_gpu_gib():
    """Bytes the driver is serving from system RAM as GPU memory, worst adapter.

    `set_per_process_memory_fraction` caps PyTorch's allocator, which is necessary but
    not sufficient: the CUDA context, cuBLAS and Triton workspaces allocate outside it,
    and on Windows WDDM anything over budget is paged to shared system memory and served
    over PCIe rather than raising. nvidia-smi does not report this, so the only honest
    check is the WDDM performance counter. A run whose shared usage climbs is measuring
    the bus, not the model, and its numbers must be thrown away.
    """
    command = ("$c = Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage' "
               "-ErrorAction SilentlyContinue; "
               "($c.CounterSamples | Measure-Object -Property CookedValue "
               "-Maximum).Maximum")
    try:
        output = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                                capture_output=True, text=True, timeout=30)
        return float(output.stdout.strip()) / 2 ** 30
    except Exception:
        return float("nan")


def chunked_head_causal_loss(hidden, head, labels, position_budget):
    """Plain causal CE with the head folded into the chunk loop.

    ``chunked_head_loss`` in distillkit does this for the sparse teacher divergences,
    whose ``fn`` signature carries target values and a mask. Pretraining from scratch
    has neither, so this is the same structure over ordinary cross-entropy: project one
    slice of the post-norm state, reduce it to a scalar, free the logits, and let
    checkpointing recompute the slice in backward. No full-vocabulary tensor is ever
    alive -- which is the difference from ``chunked_causal_lm_loss``, which chunks the
    fp32 upcast but is still handed logits the head has already materialized.

    The budget counts *total positions*, not positions per sequence: a chunk's logits
    are ``[batch, chunk, vocab]``, so a fixed per-sequence chunk silently allocates
    four times as much at batch 4.
    """
    shifted = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
    batch, seq_len = hidden.shape[0], hidden.shape[1]
    chunk = max(1, position_budget // max(1, batch))
    counted = (shifted != -100).sum()
    total = None

    def contribution(state, ids):
        logits = head(state)
        return torch.nn.functional.cross_entropy(
            logits.float().view(-1, logits.shape[-1]), ids.reshape(-1),
            ignore_index=-100, reduction="sum")

    for start in range(0, seq_len, chunk):
        stop = min(start + chunk, seq_len)
        piece = checkpoint(contribution, hidden[:, start:stop], shifted[:, start:stop],
                           use_reentrant=False, preserve_rng_state=False)
        total = piece if total is None else total + piece
    return total / torch.clamp(counted.to(total.dtype), min=1.0)


def measure(model, config, batch, length, steps, warmup, device="cuda",
            loss_kind="chunked_ce", position_budget=4096):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)

    tokens = torch.randint(0, config.vocab_size, (batch, length), device=device)
    attention = torch.ones_like(tokens)

    def step():
        optimizer.zero_grad(set_to_none=True)
        if loss_kind == "cce":
            # Never forms the logits at all: the log-sum-exp over the vocabulary is
            # reduced in SRAM, so memory is O(N + |V|) rather than O(N|V|).
            hidden = model.model(input_ids=tokens, attention_mask=attention,
                                 use_cache=False).last_hidden_state
            loss = linear_cross_entropy(hidden, model.lm_head.weight, tokens,
                                        shift=1, reduction="mean")
        elif loss_kind == "chunked_head":
            # The head never runs over the whole sequence: take the model's post-norm
            # state and fold the projection into the loss instead.
            hidden = model.model(input_ids=tokens, attention_mask=attention,
                                 use_cache=False).last_hidden_state
            loss = chunked_head_causal_loss(hidden, model.lm_head, tokens,
                                            position_budget)
        else:
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
        "shared_gib": shared_gpu_gib(),
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
    parser.add_argument("--loss", default="chunked_ce",
                        choices=("chunked_ce", "chunked_head", "cce"),
                        help="chunked_ce materializes logits then chunks the upcast; "
                             "chunked_head folds the projection into the loss loop")
    parser.add_argument("--position-budget", type=int, default=4096)
    parser.add_argument("--attn", default="sdpa",
                        choices=("sdpa", "flash_attention_2", "eager"))
    parser.add_argument("--liger", action="store_true",
                        help="swap SwiGLU and RMSNorm for Liger fused kernels")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the text model")
    parser.add_argument("--spill-tolerance-gib", type=float, default=0.25,
                        help="abort if WDDM shared GPU memory climbs this far "
                             "above baseline: the run would be measuring PCIe")
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

    if args.loss == "cce" and linear_cross_entropy is None:
        raise SystemExit("cut-cross-entropy is not installed")
    patched_linear = install_device_aware_linear_attention()
    install_expanded_gqa_attention()
    kernels = {
        "attn_implementation": args.attn,
        "flash_attn": _module_version("flash_attn"),
        "fla": _module_version("fla"),
        "causal_conv1d": _module_version("causal_conv1d"),
        "triton": _module_version("triton"),
        "cut_cross_entropy": _module_version("cut_cross_entropy"),
        "linear_attention_ops_patched": patched_linear,
        "fused_linear_attention_available": bool(fused_linear_attention_available()),
        "fused_kernel_supports_gqa": bool(fused_kernel_supports_gqa()),
        "torch_compile": bool(args.compile),
        "liger": _module_version("liger_kernel") if args.liger else None,
    }
    print(json.dumps(kernels), flush=True)

    baseline_shared = shared_gpu_gib()
    print("baseline shared GPU memory: %.2f GiB (spill guard trips at +%.2f)"
          % (baseline_shared, args.spill_tolerance_gib), flush=True)

    report = {"device": torch.cuda.get_device_name(0), "length": args.length,
              "baseline_shared_gib": baseline_shared,
              "spill_tolerance_gib": args.spill_tolerance_gib,
              "dtype": "bfloat16", "optimizer": "bnb.AdamW8bit",
              "gradient_checkpointing": not args.no_checkpointing,
              "loss": args.loss, "position_budget": args.position_budget,
              "kernels": kernels,
              "vram_fraction": args.vram_fraction,
              "total_vram_gib": torch.cuda.get_device_properties(0).total_memory / 2 ** 30,
              "configurations": {}}

    for label, hidden, layers, vocab in CONFIGURATIONS:
        if args.only and label not in args.only:
            continue
        config = build(hidden, layers, vocab, attn_implementation=args.attn)
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
        if args.liger:
            entry["liger_swapped"] = apply_liger(model, config)
            print("  liger swapped: %s" % json.dumps(entry["liger_swapped"]),
                  flush=True)
        if args.compile:
            # Only the decoder stack. The loss paths are chunked or fused already,
            # and compiling across a checkpoint boundary is what an earlier branch
            # found buys nothing here.
            model.model = torch.compile(model.model)
        for batch in args.batches:
            try:
                row = measure(model, config, batch, args.length, args.steps,
                              args.warmup, loss_kind=args.loss,
                              position_budget=args.position_budget)
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
            spilled = row["shared_gib"] - baseline_shared
            row["shared_delta_gib"] = spilled
            entry["runs"].append(row)
            if spilled > args.spill_tolerance_gib:
                print("  batch %-3d SPILLED: shared GPU memory +%.2f GiB above "
                      "baseline. Result discarded, sweep stopped."
                      % (batch, spilled), flush=True)
                row["spilled"] = True
                entry["spilled_at_batch"] = batch
                break
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
