"""Gate 5, in the form this box can actually run: stage-1 sidecar training on one GPU.

Spec §5.5 wants a smoke run proving shapes, finite losses, memory fit and a
tokens/sec figure to extrapolate from. It assumes DeepSpeed ZeRO-2 with optimizer
offload, which does not exist here: deepspeed is not installed and this Windows
torch build reports NCCL unavailable, so multi-GPU ZeRO needs WSL2/Linux.

Stage 1 does not need it. With the backbone frozen only the sidecar trains -- on
the order of 66M parameters against the student's 4.27B -- so grads and optimizer
state are under a gigabyte and the whole step fits on a single 24 GB card:

    bf16 weights  4.27B x 2      = 8.5 GB
    trainable     ~66M           = grads + AdamW/Muon state < 1.0 GB
    activations   with checkpointing, scales with batch x seq

That is also exactly the §5.6 pilot configuration (backbone frozen, linear probe
on frozen features), so what this measures is the arm that runs next.

What it reports, and why each matters:
  * peak VRAM per phase, so the batch/seq that fits is measured not guessed
  * tokens/sec, to extrapolate the 1M and 5M token runs
  * gather time as a fraction of step time -- the open `training_overlap_verified`
    question. The isolated benchmark said ~70 ms at batch 8 x 4096; what matters
    is that figure *next to a real step*, under real memory pressure.
  * loss finite and gradients reaching W_side_proj, since a zero-init projection
    that never moves is the failure this whole design risks.

Run only when the GPUs are free; it loads an 8.5 GB model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_GGUF = os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "models--unsloth--Qwen3.8-Flash-Next-GGUF", "snapshots",
    "38bb39ee97821de2c9009abb7e93950eec396e66", "UD-IQ4_XS",
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf",
)
DEFAULT_STUDENT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "student-hf")
)


def gb(n: int) -> float:
    return n / 1024**3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", default=DEFAULT_GGUF)
    parser.add_argument("--student", default=DEFAULT_STUDENT)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=6, help="timed steps after warmup")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resident", action="store_true",
                        help="copy the 28.8 GB table into RAM; otherwise mmap (cold faults)")
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA available"); return 2
    device = torch.device(args.device)
    free, total = torch.cuda.mem_get_info(device)
    print(f"== {torch.cuda.get_device_name(device)}  {gb(free):.1f} GB free / {gb(total):.1f} GB")
    if gb(free) < 12:
        print(f"ERROR: need ~12 GB free, have {gb(free):.1f} GB. Something else holds VRAM.")
        return 2

    from transformers import AutoTokenizer

    from distillkit.chunked_ce import chunked_causal_lm_loss
    from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM
    from distillkit.ngram_hash import NGramHasher
    from distillkit.ngram_table import GGUFNGramTable
    from distillkit.optimizers import build_mixed_optimizer

    results: dict[str, object] = {
        "device": torch.cuda.get_device_name(device),
        "batch": args.batch, "seq": args.seq,
        "gradient_checkpointing": not args.no_checkpointing,
        "table_resident": args.resident,
    }

    table = GGUFNGramTable(args.gguf)
    print("==", table)
    if args.resident:
        print(f"   resident load {table.load_resident():.1f}s")
    hasher = NGramHasher()

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    # flash_attn has no Windows wheels; dtype must be set explicitly since that flag
    # is normally what selects bfloat16.
    model = Qwen35SidecarForCausalLM.from_pretrained(args.student, dtype=torch.bfloat16)
    model.to(device)
    load_s = time.perf_counter() - t0
    weights_gb = gb(torch.cuda.max_memory_allocated(device))
    print(f"   student on GPU in {load_s:.1f}s, weights {weights_gb:.2f} GB")

    model.loss_function = chunked_causal_lm_loss  # as DistillationTrainer does
    model.freeze_backbone()
    if not args.no_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for _, p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"   stage-1 trainable: {len(trainable)} tensors, {n_trainable/1e6:.1f}M "
          f"of {n_total/1e9:.2f}B ({100*n_trainable/n_total:.3f}%)")
    results |= {"trainable_tensors": len(trainable), "trainable_params": n_trainable,
                "total_params": n_total, "weights_gb": weights_gb}

    optimizer = build_mixed_optimizer(model, lr=1e-4)
    print(f"   optimizer: {type(optimizer).__name__}, "
          f"{sum(len(g['params']) for g in optimizer.param_groups)} params in "
          f"{len(optimizer.param_groups)} groups")

    generator = torch.Generator().manual_seed(0)
    vocab = min(hasher.config.vocab_size, model.config.vocab_size)

    def make_batch():
        ids = torch.randint(0, vocab, (args.batch, args.seq), generator=generator, dtype=torch.long)
        t_hash = time.perf_counter()
        rows = hasher.row_indices(ids)
        raw = torch.from_numpy(table.gather_raw(rows).copy())
        gather_s = time.perf_counter() - t_hash
        return ids, raw, gather_s

    side_proj = model.model.layers[model.config.sidecar_layer_index].sidecar.W_side_proj.weight

    def step(ids, raw):
        out = model(
            input_ids=ids.to(device, non_blocking=True),
            attention_mask=torch.ones_like(ids).to(device, non_blocking=True),
            labels=ids.to(device, non_blocking=True),
            ngram_raw=raw.to(device, non_blocking=True),
            sidecar_enabled=True,
            return_dict=True,
        )
        out.loss.backward()
        # Read the gradient here: zero_grad(set_to_none=True) below drops it, and
        # reading after the step reports None as 0.0 -- indistinguishable from the
        # dead-projection failure this is supposed to detect.
        grad = side_proj.grad.float().norm().item() if side_proj.grad is not None else None
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return out.loss.detach(), grad

    print("\n   warmup...")
    ids, raw, _ = make_batch()
    loss, _ = step(ids, raw)
    torch.cuda.synchronize(device)
    assert torch.isfinite(loss), "warmup loss is not finite"
    torch.cuda.reset_peak_memory_stats(device)

    gathers, steps_s, losses, grads = [], [], [], []
    for i in range(args.steps):
        ids, raw, gather_s = make_batch()
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        loss, grad = step(ids, raw)
        torch.cuda.synchronize(device)
        steps_s.append(time.perf_counter() - t0)
        gathers.append(gather_s)
        losses.append(loss.item())
        grads.append(grad)
        print(f"   step {i}: {steps_s[-1]*1000:7.1f} ms  gather {gather_s*1000:6.1f} ms  "
              f"loss {losses[-1]:.4f}")

    peak_gb = gb(torch.cuda.max_memory_allocated(device))
    reserved_gb = gb(torch.cuda.max_memory_reserved(device))
    med_step = sorted(steps_s)[len(steps_s) // 2]
    med_gather = sorted(gathers)[len(gathers) // 2]
    tokens = args.batch * args.seq
    tps = tokens / med_step

    assert all(g is not None and g > 0 for g in grads), (
        f"W_side_proj received no gradient on some step: {grads} -- a zero-init "
        "projection with no gradient would stay zero for the whole run"
    )

    print(f"\n== results ==")
    print(f"  peak allocated {peak_gb:.2f} GB   reserved {reserved_gb:.2f} GB "
          f"(card {gb(total):.1f} GB)")
    print(f"  median step {med_step*1000:.1f} ms  -> {tps:,.0f} tok/s")
    print(f"  median gather {med_gather*1000:.1f} ms = {100*med_gather/med_step:.1f}% of a step")
    print(f"  losses finite: {all(l == l for l in losses)}  range "
          f"{min(losses):.4f}..{max(losses):.4f}")
    print(f"  W_side_proj grad norm per step: "
          f"{min(grads):.4f}..{max(grads):.4f} (all nonzero)")
    for n_tokens, label in ((1e6, "1M smoke"), (5e6, "5M pilot")):
        print(f"  extrapolated {label}: {n_tokens/tps/3600:.2f} h")

    results |= {
        "peak_allocated_gb": peak_gb, "peak_reserved_gb": reserved_gb,
        "card_total_gb": gb(total), "median_step_s": med_step,
        "median_gather_s": med_gather, "gather_fraction_of_step": med_gather / med_step,
        "tokens_per_s": tps, "losses": losses, "w_side_proj_grad_norms": grads,
        "hours_1m_tokens": 1e6 / tps / 3600, "hours_5m_tokens": 5e6 / tps / 3600,
    }
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print("wrote", args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
