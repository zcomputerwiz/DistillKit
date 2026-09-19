"""Two-card training for the active gated residual, fused as far as this stack allows.

`smoke_train.py` runs one card and one model. This is the same experiment on both, for
the configuration the measurements landed on: blend on, `fused` norms and write,
micro-batch 32 at an effective batch of 64, and gradient checkpointing when the memory is
wanted back.

**Why the bodies split and the vocabulary does not.** `distillkit.parallel` shards MLPs,
attention and the gated delta rule by channel and head, replicating each block's input to
both cards and reducing back to home, so the residual stream -- both layer norms, the
final norm, rotary, and the four-stream route -- stays on card 0 and the outer model is
untouched. `shard_model` also splits the tied embedding by vocabulary rows, which is the
right call at the 248,320-token vocabulary it was written for, where that one parameter is
0.636B. It is the wrong call here: at vocabulary 16,384 and hidden 512 the tie is 8.4M
against a 37M body, and splitting it costs the one thing worth more than the balance --
`linear_cross_entropy` never forms the logits, and a vocabulary-parallel loss has to. Cut
Cross-Entropy does ship `VocabParallelOptions`, but it wants a `torch.distributed`
process group, and this design has neither: NCCL is absent from this wheel and both cards
are in one process. So the head stays whole on home and the loss is unchanged.

**What the peer collectives are.** Not `torch.distributed`. Both devices are in this
process and CUDA peer access lets either read the other's memory, so an all-reduce is a
peer copy and an add, and autograd differentiates the copy without help. Measured on this
machine at 38-48 GB/s for 64 MiB against NCCL's 37.5.

**What this buys, measured, and where it stops.** At hidden 512 over ten layers with the
route active, micro-batch 32: one card 57,507 tokens per second at 15.96 GiB, two cards
60,537 at 12.44 and 3.55. So the split is worth 5.3% and moves 3.5 GiB off the home card,
which is worth having and is not what makes a bigger model fit -- checkpointing is. Two
cards without it still run out of memory at twenty layers, exactly as one card does.

The split is also not free at every shape. Checkpointed at micro-batch 8 it *loses*:
28,452 tokens per second against one card's 37,531. Halving an already small GEMM leaves
the launch and the reduction dominating, and checkpointing doubles the forward that pays
them. Split when the micro-batch is large; do not split a small one.

**The autotuner race, and why the warm-up step exists.** Sharding used to fail with
`TypeError: 'NoneType' object is not a mapping` out of Triton's `check_disk_cache`. It is
not a cache problem and not a size problem, though it impersonated both: Triton's
`Autotuner` holds the arguments it is benchmarking on the instance, in `self.nargs`, and
autograd gives each device its own backward thread, so a sharded model puts two threads
through one module-level autotuner and one clears `nargs` while the other is still
benchmarking. It only fires on a cold autotune, which is why running the same shape twice
appeared to cure it. `warm_autotune` runs one step with the extra threads off, and after
that there is nothing left to race over.

    CUDA_VISIBLE_DEVICES=0,1 python scratch/dense_gr/tp_train.py --blend 1.0
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

_original = metadata.version


def _version(name):
    try:
        return _original(name)
    except metadata.PackageNotFoundError:
        if name == "triton":
            return _original("triton-windows")
        raise


metadata.version = _version

import numpy as np  # noqa: E402
import torch  # noqa: E402

from benchmark import (ATTENTION_RATIOS, SpillWatch, apply_liger, build,  # noqa: E402
                       variant_tag)
from copy_probe import copy_probe, format_probe  # noqa: E402
from cut_cross_entropy import linear_cross_entropy  # noqa: E402
from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402
from distillkit.parallel import (clip_grad_norm, peer_capable,  # noqa: E402
                                 sharded_parameter_report,
                                 sync_replicated_gradients)
from distillkit.parallel.blocks import (TensorParallelAttention,  # noqa: E402
                                        TensorParallelMLP)
from vocab_remap import build_vocabulary, cached_remap  # noqa: E402

BASE = "D:/DeepThought/Projects/HybridModel/student-2b-hf"
STORE = Path("scratch/code_training/tokens-v2")


def warm_autotune(model, home, length, micro):
    """Run one step single-threaded so Triton's autotuner never races itself.

    `triton.runtime.Autotuner` keeps the arguments it is benchmarking in `self.nargs`, on
    the instance, and clears them when its `run` returns. Autograd gives each device its
    own backward thread, so sharding puts two threads through the same module-level
    autotuner at once: one clears `nargs` while the other is still inside `benchmark()`,
    which then reads `None` and raises `TypeError: 'NoneType' object is not a mapping`
    out of Triton's own `check_disk_cache`.

    It only bites on a cold autotune, because a cached config never benchmarks -- which
    is why it looked intermittent, looked size-dependent, and disappeared whenever the
    same shape had been run before. Nothing about it is ours to fix upstream from here.

    Turning the extra threads off for exactly one step is enough: after that every kernel
    this configuration uses has a config in the autotuner's own dictionary, and the
    threads have nothing left to race over. The alternative, a lock around every launch,
    would serialize the two cards for the whole run to protect a few seconds of warm-up.
    """
    tokens = torch.randint(0, model.config.vocab_size, (micro, length), device=home)
    torch.autograd.set_multithreading_enabled(False)
    try:
        hidden = model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                             use_cache=False).last_hidden_state
        linear_cross_entropy(hidden, model.lm_head.weight, tokens, shift=1,
                             reduction="mean").backward()
    finally:
        torch.autograd.set_multithreading_enabled(True)
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()


def shard_bodies(model, devices):
    """Split every layer's blocks across the cards, leaving the tie whole on home.

    `shard_model` is the same loop plus `_shard_tied_embeddings`; this is that loop
    without it, so `lm_head` stays a single dense weight on home and the loss stays
    `linear_cross_entropy`.
    """
    from distillkit.models.qwen35.tp_gated_delta_module import TensorParallelGatedDeltaNet

    resolved = [torch.device(d) for d in devices]
    if len(resolved) < 2:
        raise SystemExit("tensor parallelism needs at least two devices")
    if not peer_capable(resolved):
        raise SystemExit(
            "peer access is unavailable between these devices; every reduction would "
            "stage through host memory and the split would cost more than it saves")
    model.to(resolved[0])
    base = getattr(model, "model", model)
    counts = {"mlp": 0, "full_attention": 0, "linear_attention": 0}
    for layer in base.layers:
        layer.mlp = TensorParallelMLP(layer.mlp, resolved)
        counts["mlp"] += 1
        attention = getattr(layer, "self_attn", None)
        if attention is not None:
            layer.self_attn = TensorParallelAttention(attention, resolved)
            counts["full_attention"] += 1
        linear = getattr(layer, "linear_attn", None)
        if linear is not None:
            layer.linear_attn = TensorParallelGatedDeltaNet(linear, resolved)
            counts["linear_attention"] += 1
    model.config.use_cache = False
    model._distillkit_tp_devices = tuple(str(d) for d in resolved)
    base._distillkit_tp_devices = model._distillkit_tp_devices
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--vocab", type=int, default=16_384)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--ratio", default="3:1", choices=sorted(ATTENTION_RATIOS))
    parser.add_argument("--blend", type=float, default=1.0,
                        help="gated residual strength; this trainer exists for the "
                             "active route, so it defaults on")
    parser.add_argument("--norm-mode", default="fused",
                        choices=("exact", "fast", "fused"),
                        help="fused is 15.5%% faster and 0.0012 nats from exact over "
                             "2,000 steps; exact is required to convert a donor")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--accumulate", type=int, default=2,
                        help="micro-batches per step. 2 puts the micro-batch at 32, "
                             "which measured fastest for the active route")
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--tokens", type=int, default=524_288_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpointing", action="store_true",
                        help="recompute activations; 4x less memory for 26%% more time, "
                             "and what lets a wider model fit at all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--evaluate-every", type=int, default=1_000)
    parser.add_argument("--evaluate-windows", type=int, default=64)
    parser.add_argument("--probe-every", type=int, default=500)
    parser.add_argument("--probe-half", type=int, default=256)
    parser.add_argument("--probe-windows", type=int, default=64)
    parser.add_argument("--spill-every", type=float, default=30.0)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--checkpoints", type=Path,
                        default=Path("scratch/dense_gr/checkpoints-tp"))
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.batch % args.accumulate:
        raise SystemExit("--batch %d is not divisible by --accumulate %d; the micro-"
                         "batches would be uneven and the loss mis-weighted"
                         % (args.batch, args.accumulate))
    variant = variant_tag(args.ratio, args.blend, args.norm_mode, args.seed)
    if args.output is None:
        args.output = Path("scratch/dense_gr/tp-%s.json" % variant)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    home = torch.device(args.devices[0])
    torch.cuda.set_per_process_memory_fraction(0.90, home.index or 0)
    started = time.perf_counter()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    counts = np.load(args.store / "train-counts.npy")
    kept, forward, bytes_ids = build_vocabulary(counts, tokenizer, args.vocab)
    coverage = float(counts[np.asarray(kept)].sum() / counts.sum())
    stream, original_tokens, compact_tokens = cached_remap(
        args.store, "train", args.vocab, tokenizer, forward, bytes_ids, counts, kept)
    stream = np.asarray(stream)
    inflation = compact_tokens / original_tokens
    print("vocabulary: kept %d ids, %.4f%% coverage, inflation %.4f"
          % (len(kept), 100 * coverage, inflation), flush=True)

    config = build(args.hidden, args.layers, args.vocab, ratio=args.ratio,
                   blend=args.blend, attn_implementation="flash_attention_2")
    config.residual_stream_norm_mode = args.norm_mode
    torch.manual_seed(args.seed)
    model = Qwen35WidenedForCausalLM(config).to(dtype=torch.bfloat16)
    swapped = apply_liger(model, config)
    split = shard_bodies(model, args.devices)
    model.model.gradient_checkpointing = bool(args.checkpointing)
    report = sharded_parameter_report(model)
    print("tensor parallel over %s: %d MLPs, %d attention, %d gated-delta; %.1f%% of "
          "parameters split" % (args.devices, split["mlp"], split["full_attention"],
                                split["linear_attention"],
                                100 * report["sharded_fraction"]), flush=True)
    print("model: %.1fM parameters, liger %s, checkpointing %s, norm mode %s"
          % (sum(p.numel() for p in model.parameters()) / 1e6, json.dumps(swapped),
             bool(args.checkpointing), args.norm_mode), flush=True)

    held_stream, held_original, held_compact = cached_remap(
        args.store, "calibration", args.vocab, tokenizer, forward, bytes_ids, counts,
        kept, verbose=False)
    held_rng = np.random.default_rng(12345)
    held_starts = held_rng.integers(0, held_stream.shape[0] - args.length - 1,
                                    size=args.evaluate_windows)
    evaluation = torch.from_numpy(
        np.stack([held_stream[s:s + args.length] for s in held_starts]).astype(np.int64))

    warm_autotune(model, home, args.length, args.batch // args.accumulate)
    print("autotune warmed on one thread", flush=True)

    parameters = [p for p in model.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(parameters, lr=args.lr)
    except Exception:
        optimizer = torch.optim.AdamW(parameters, lr=args.lr)

    def evaluate():
        model.eval()
        total, seen = 0.0, 0
        with torch.no_grad():
            for start in range(0, evaluation.shape[0], args.batch // args.accumulate):
                chunk = evaluation[start:start + args.batch // args.accumulate].to(home)
                hidden = model.model(input_ids=chunk,
                                     attention_mask=torch.ones_like(chunk),
                                     use_cache=False).last_hidden_state
                loss = linear_cross_entropy(hidden, model.lm_head.weight, chunk,
                                            shift=1, reduction="mean")
                total += float(loss) * chunk.shape[0]
                seen += chunk.shape[0]
        model.train()
        return total / seen

    window = args.batch * args.length
    steps = max(1, args.tokens // window)
    generator = np.random.default_rng(args.seed)
    spill = SpillWatch(interval=args.spill_every).start()
    history = []
    model.train()
    torch.cuda.synchronize()
    train_started = time.perf_counter()

    for step in range(steps):
        starts = generator.integers(0, stream.shape[0] - args.length - 1,
                                    size=args.batch)
        batch = np.stack([stream[s:s + args.length] for s in starts]).astype(np.int64)
        tokens = torch.from_numpy(batch).to(home, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = 0.0
        for chunk in tokens.chunk(args.accumulate):
            hidden = model.model(input_ids=chunk,
                                 attention_mask=torch.ones_like(chunk),
                                 use_cache=False).last_hidden_state
            part = linear_cross_entropy(hidden, model.lm_head.weight, chunk, shift=1,
                                        reduction="mean") / args.accumulate
            part.backward()
            loss = loss + part.detach()
        # Replicated parameters -- the norms -- see only their card's share of the loss,
        # so each holds a partial gradient until these are summed.
        sync_replicated_gradients(model)
        clip_grad_norm(model, 1.0)
        optimizer.step()
        if spill.breached():
            raise SystemExit("spilled %.2f GiB into system RAM at step %d"
                             % (spill.tripped_at, step))

        if step % args.report_every == 0 or step == steps - 1:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - train_started
            seen = (step + 1) * window
            row = {"step": step, "tokens": seen, "loss": float(loss),
                   "loss_per_original_token": float(loss) * inflation,
                   "tokens_per_second": seen / elapsed,
                   "shared_delta_gib": spill.drift()}
            if step % args.evaluate_every == 0 or step == steps - 1:
                row["heldout"] = evaluate()
                row["heldout_per_original_token"] = row["heldout"] * inflation
            if args.probe_every and (step % args.probe_every == 0 or step == steps - 1):
                row["copy"] = copy_probe(model, args.vocab, half=args.probe_half,
                                         windows=args.probe_windows,
                                         batch=args.batch // args.accumulate)
            row["gib_per_device"] = [
                torch.cuda.max_memory_allocated(torch.device(d).index or 0) / 2 ** 30
                for d in args.devices]
            history.append(row)
            print("step %5d  train %7.4f  held %8s  %8.0f tok/s  %s"
                  % (step, row["loss"],
                     "%.4f" % row["heldout"] if "heldout" in row else "-",
                     row["tokens_per_second"],
                     " ".join("%.2f" % g for g in row["gib_per_device"])), flush=True)
            if "copy" in row:
                print("            %s" % format_probe(row["copy"]), flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - train_started
    result = {
        "variant": variant, "devices": list(args.devices),
        "architecture": {"ratio": args.ratio, "blend": args.blend,
                         "norm_mode": args.norm_mode, "hidden": args.hidden,
                         "layers": args.layers,
                         "checkpointing": bool(args.checkpointing),
                         "tensor_parallel": split,
                         "sharded_fraction": report["sharded_fraction"]},
        "vocab": args.vocab, "coverage": coverage, "inflation": inflation,
        "batch": args.batch, "accumulate": args.accumulate, "length": args.length,
        "steps": steps, "scored_tokens": steps * window, "seed": args.seed,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "seconds": elapsed, "tokens_per_second": steps * window / elapsed,
        "final_loss": history[-1]["loss"], "final_heldout": history[-1].get("heldout"),
        "final_copy": history[-1].get("copy"),
        "gib_per_device": history[-1]["gib_per_device"],
        "shared_delta_gib": spill.stop(),
        "setup_seconds": train_started - started, "history": history,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote %s" % args.output, flush=True)

    if not args.no_checkpoint:
        # The shards live on two cards, so the state dict has to be reunified before it
        # means anything to a loader that has not been told about the split.
        from distillkit.parallel import consolidated_state_dict
        target = args.checkpoints / ("tp-%s" % variant)
        if target.exists() and any(target.iterdir()):
            raise SystemExit("%s already holds a checkpoint" % target)
        target.mkdir(parents=True, exist_ok=True)
        torch.save(consolidated_state_dict(model), target / "model.pt")
        config.save_pretrained(target)
        tokenizer.save_pretrained(target)
        (target / "milestone.json").write_text(
            json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2),
            encoding="utf-8")
        print("CHECKPOINT %s" % target.name, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
