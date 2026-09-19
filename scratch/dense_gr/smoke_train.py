"""End-to-end smoke test: real tokens, remapped vocabulary, the measured configuration.

Everything so far has been benchmarked on random token ids, which exercises the kernels
but not the pipeline. This runs the smallest configuration on the actual Python token
store with the whole stack the substrate document recommends, and checks the things that
would quietly be wrong: that the vocabulary remap round-trips, that the loss falls, that
throughput matches what the benchmark promised, and that nothing spilled.

The remap is the design from `docs/dense_gr.md`: the original tokenizer stays the only
thing that touches text, a bijection carries kept ids to a compact space, and ids below
the cut decompose into their byte tokens rather than an UNK, so the mapping is lossless.
All 256 byte tokens and every special are kept regardless of frequency -- without the byte
tokens the fallback has holes and the "lossless" claim is false.

    CUDA_VISIBLE_DEVICES=0 python scratch/dense_gr/smoke_train.py
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
from distillkit.models.qwen35.csa2 import routing_report  # noqa: E402
from vocab_remap import (bytes_to_unicode, build_vocabulary,  # noqa: E402,F401
                         byte_token_ids, cached_remap)

BASE = "D:/DeepThought/Projects/HybridModel/student-2b-hf"
STORE = Path("scratch/code_training/tokens-v2")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=int, default=16_384)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--ratio", default="1:1", choices=sorted(ATTENTION_RATIOS),
                        help="gated-delta-net layers per full-attention layer; "
                             "1:1 is what these arms have run, 3:1 is Qwen3-Next's")
    parser.add_argument("--blend", type=float, default=0.0,
                        help="gated residual route strength; 0 leaves it inert, "
                             "which is what every arm so far has run")
    parser.add_argument("--norm-mode", default="exact",
                        choices=("exact", "fast", "fused"),
                        help="how the branch read normalises; exact is "
                             "bit-identical to the stock norm and what a "
                             "converted model needs, fused is fastest and "
                             "loses about one bf16 ulp")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--accumulate", type=int, default=1,
                        help="micro-batches per optimizer step; the effective batch "
                             "stays --batch, so a blend comparison compares blend "
                             "and not batch size")
    parser.add_argument("--tokens", type=int, default=30_000_000,
                        help="scored tokens; one pass over the v1 store is 30.7M")
    parser.add_argument("--passes", type=float, default=None,
                        help="passes over the corpus, which overrides --tokens. This is "
                             "the fair budget across vocabularies: equal scored tokens "
                             "would give a small vocabulary less text for the same count")
    parser.add_argument("--evaluate-every", type=int, default=0,
                        help="steps between held-out evaluations; 0 disables")
    parser.add_argument("--evaluate-windows", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--spill-every", type=float, default=30.0,
                        help="seconds between background spill checks; the counter\n"
                             "read costs 1.8 s, so it is polled off the training thread")
    parser.add_argument("--seed", type=int, default=0,
                        help="seeds model init and the data order. The probe is seeded "
                             "separately and identically for every run, so arms and "
                             "seeds are scored on the same sequences")
    parser.add_argument("--probe-every", type=int, default=0,
                        help="steps between copy probes; 0 disables")
    parser.add_argument("--probe-half", type=int, default=256,
                        help="tokens in the block that gets repeated")
    parser.add_argument("--probe-windows", type=int, default=64)
    parser.add_argument("--mla", action="store_true",
                        help="replace the full-attention layers' key/value side with a "
                             "compressed latent")
    parser.add_argument("--csa2", action="store_true",
                        help="route the latent attention through Full/Reindex/Reuse "
                             "modes; requires --mla")
    parser.add_argument("--csa2-modes", nargs="+",
                        default=["full", "reuse", "full", "reindex", "reuse"],
                        help="one mode per full-attention layer")
    parser.add_argument("--csa2-top-k", type=int, default=256)
    parser.add_argument("--csa2-local-window", type=int, default=128)
    parser.add_argument("--csa2-block-size", type=int, default=128)
    parser.add_argument("--mla-latent-dim", type=int, default=128)
    parser.add_argument("--checkpoints", type=Path,
                        default=Path("scratch/dense_gr/checkpoints-smoke"),
                        help="root for the end-of-run checkpoint")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("scratch/dense_gr/smoke-train.json"))
    args = parser.parse_args()
    variant = variant_tag(args.ratio, args.blend, args.norm_mode, args.seed)
    if args.mla:
        variant += "-csa2" if args.csa2 else "-mla"
    if args.output == Path("scratch/dense_gr/smoke-train.json"):
        args.output = Path("scratch/dense_gr/smoke-train-%s.json" % variant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.accumulate < 1 or args.batch % args.accumulate:
        # `chunk` splits unevenly when it does not divide, and the loop divides every
        # micro-batch's mean by the same `accumulate` -- so an uneven split silently
        # weights some tokens more than others and the reported loss is not the batch's.
        raise SystemExit("--batch %d is not divisible by --accumulate %d; the micro-"
                         "batches would be uneven and the loss would be mis-weighted"
                         % (args.batch, args.accumulate))
    store = args.store

    torch.cuda.set_per_process_memory_fraction(0.90, 0)
    started = time.perf_counter()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)

    raw = np.memmap(store / "train.bin", dtype=np.uint32, mode="r")
    print("store: %d tokens" % raw.shape[0], flush=True)

    # Counting 3B ids takes a couple of minutes and the answer never changes, so the
    # corpus ships the ranking beside the tokens.
    cache = store / "train-counts.npy"
    if cache.exists():
        counts = np.load(cache)
        print("counts: loaded %s" % cache.name, flush=True)
    else:
        counts = np.bincount(np.asarray(raw, dtype=np.int64), minlength=248_320)
    kept, forward, bytes_ids = build_vocabulary(counts, tokenizer, args.vocab)
    coverage = float(counts[np.asarray(kept)].sum() / counts.sum())
    print("vocabulary: kept %d ids, %.4f%% coverage" % (len(kept), 100 * coverage),
          flush=True)

    stream, original_tokens, compact_tokens = cached_remap(
        store, "train", args.vocab, tokenizer, forward, bytes_ids, counts, kept)
    # Into RAM: the training loop draws random windows, and a cold page cache over a
    # multi-gigabyte file would pay for that at every step of the first pass.
    stream = np.asarray(stream)
    print("remap: %d tokens -> %d, inflation %.4f, %.1f GiB resident"
          % (original_tokens, compact_tokens, compact_tokens / original_tokens,
             stream.nbytes / 2 ** 30), flush=True)

    # Round trip: the compact stream must decode to what the original ids decode to.
    inverse = np.asarray(kept, dtype=np.int64)
    sample = stream[:4096].astype(np.int64)
    round_tripped = tokenizer.decode(inverse[sample].tolist())
    reference = tokenizer.decode(np.asarray(raw[:4096]).tolist())
    matches = round_tripped[:2000] == reference[:2000]
    print("round trip on the first 4096 compact tokens: %s" % matches, flush=True)

    config = build(args.hidden, args.layers, args.vocab, ratio=args.ratio,
                   blend=args.blend,
                   attn_implementation="flash_attention_2")
    config.residual_stream_norm_mode = args.norm_mode
    if args.mla:
        config.mla_enabled = True
        config.mla_latent_dim = args.mla_latent_dim
    if args.csa2:
        config.csa2_enabled = True
        config.csa2_modes = list(args.csa2_modes)
        config.csa2_top_k = args.csa2_top_k
        config.csa2_local_window = args.csa2_local_window
        config.csa2_block_size = args.csa2_block_size
    torch.manual_seed(args.seed)
    model = Qwen35WidenedForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    model.train()
    swapped = apply_liger(model, config)
    parameters = sum(p.numel() for p in model.parameters())
    print("model: %.1fM parameters, liger %s" % (parameters / 1e6, json.dumps(swapped)),
          flush=True)

    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                    weight_decay=0.1)
    # Polled on a daemon thread: reading it inline costs 1.81 s per report and
    # leaves the GPU with nothing queued for all of it.
    spill = SpillWatch(interval=args.spill_every).start()

    # Held-out: the calibration split shares no repository with train, so this measures
    # generalization rather than how much of the corpus has been memorized -- which is
    # what training loss becomes once a budget spans several passes.
    evaluation = None
    if args.evaluate_every:
        held_stream, held_original, held_compact = cached_remap(
            store, "calibration", args.vocab, tokenizer, forward, bytes_ids, counts, kept)
        rng = np.random.default_rng(12345)
        starts = rng.integers(0, held_stream.shape[0] - args.length - 1,
                              size=args.evaluate_windows)
        evaluation = torch.from_numpy(
            np.stack([held_stream[s:s + args.length] for s in starts]).astype(np.int64))
        print("held-out: %d tokens -> %d, %d fixed windows"
              % (held_original, held_compact, args.evaluate_windows), flush=True)

    @torch.no_grad()
    def evaluate():
        model.eval()
        total, batches = 0.0, 0
        for start in range(0, evaluation.shape[0], args.batch):
            chunk = evaluation[start:start + args.batch].to("cuda")
            state = model.model(input_ids=chunk,
                                attention_mask=torch.ones_like(chunk),
                                use_cache=False).last_hidden_state
            total += float(linear_cross_entropy(state, model.lm_head.weight, chunk,
                                                shift=1, reduction="mean"))
            batches += 1
        model.train()
        return total / max(batches, 1)

    # Nats per *original* token, so vocabularies are comparable: a cut that expands more
    # tokens is charged for the expansion rather than rewarded with an easier softmax.
    inflation = compact_tokens / original_tokens

    window = args.batch * args.length
    if args.passes is not None:
        args.tokens = int(args.passes * stream.shape[0])
    steps = max(1, args.tokens // window)
    generator = np.random.default_rng(args.seed)
    history = []
    torch.cuda.synchronize()
    train_started = time.perf_counter()
    for step in range(steps):
        starts = generator.integers(0, stream.shape[0] - args.length - 1,
                                    size=args.batch)
        batch = np.stack([stream[s:s + args.length] for s in starts]).astype(np.int64)
        tokens = torch.from_numpy(batch).to("cuda", non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        # One micro-batch is the ordinary path and stays bitwise what it was. More than
        # one exists because an active gated residual does not fit otherwise: at blend 0
        # the route short-circuits, and at blend 1 it materializes the four-branch read
        # and write, which runs out of 24 GiB at batch 64. Accumulating keeps the
        # *effective* batch identical across arms, so a blend comparison is a comparison
        # of blend rather than of batch size.
        loss = 0.0
        for chunk in tokens.chunk(args.accumulate):
            hidden = model.model(input_ids=chunk,
                                 attention_mask=torch.ones_like(chunk),
                                 use_cache=False).last_hidden_state
            part = linear_cross_entropy(hidden, model.lm_head.weight, chunk, shift=1,
                                        reduction="mean") / args.accumulate
            part.backward()
            loss = loss + part.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if spill.breached():
            # Set by the watcher thread the moment a poll exceeded the tolerance, so the
            # step this stops on is the first one after the breach rather than the next
            # multiple of report_every.
            args.output.write_text(json.dumps(
                {"aborted": "spilled to system RAM",
                 "shared_delta_gib": spill.tripped_at, 
                 "step": step, "history": history}, indent=2), encoding="utf-8")
            raise SystemExit("spilled %.2f GiB into system RAM at step %d; stopping"
                             % (spill.tripped_at, step))
        if step % args.report_every == 0 or step == steps - 1:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - train_started
            seen = (step + 1) * window
            row = {"step": step, "tokens": seen, "passes": seen / stream.shape[0],
                   "loss": float(loss), "loss_per_original_token": float(loss) * inflation,
                   "tokens_per_second": seen / elapsed}
            # Read before the probe and the held-out pass. Both run their own forward at
            # their own sequence length, and the layers keep only the last routing they
            # computed -- reading after them reports the probe's 512-token routing, where
            # every block is reachable and the density is trivially 1.00, instead of the
            # training batch's.
            routing = routing_report(model)
            if routing:
                row["routing"] = routing
            if evaluation is not None and (step % args.evaluate_every == 0
                                           or step == steps - 1):
                row["heldout"] = evaluate()
                row["heldout_per_original_token"] = row["heldout"] * inflation
            if args.probe_every and (step % args.probe_every == 0 or step == steps - 1):
                row["copy"] = copy_probe(model, args.vocab, half=args.probe_half,
                                         windows=args.probe_windows, batch=args.batch)
            history.append(row)
            print("step %5d  %5.2f passes  train %7.4f  held %8s  norm %7.4f  %8.0f tok/s"
                  % (step, row["passes"], row["loss"],
                     "%.4f" % row["heldout"] if "heldout" in row else "-",
                     row.get("heldout_per_original_token",
                             row["loss_per_original_token"]),
                     row["tokens_per_second"]), flush=True)
            if "copy" in row:
                print("            %s" % format_probe(row["copy"]), flush=True)
            # Windows pages CUDA allocations out to system RAM over PCIe instead of
            # raising OOM, so a spilled run keeps reporting 100% GPU utilization while
            # throughput collapses. Stop on it rather than discovering it in the summary
            # of a run that has already burned hours.
            row["shared_delta_gib"] = spill.drift()
            # Loss alone cannot tell a router that learned to route from one that
            # collapsed onto the blocks the local window opens for free.
            if routing:
                print("            routing " + "  ".join(
                    "L%d %s d%.2f s%.2f h%.2f" % (r["layer"], r["mode"][:3],
                                                  r["density"], r["selected"],
                                                  r["entropy"]) for r in routing),
                      flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - train_started
    spilled = spill.stop()
    report = {
        "vocab": args.vocab, "kept_ids": len(kept), "coverage": coverage,
        "store": str(store), "seed": args.seed,
        "architecture": {"ratio": args.ratio, "blend": args.blend,
                         "variant": variant, "hidden": args.hidden,
                         "norm_mode": args.norm_mode,
                         "layers": args.layers,
                         "full_attention_layers": [
                             index for index, kind in enumerate(config.layer_types)
                             if "linear" not in str(kind)],
                         "mla": bool(getattr(config, "mla_enabled", False)),
                         "csa2": bool(getattr(config, "csa2_enabled", False)),
                         "csa2_modes": list(args.csa2_modes) if args.csa2 else None,
                         "csa2_top_k": args.csa2_top_k if args.csa2 else None,
                         "csa2_local_window": (args.csa2_local_window if args.csa2
                                               else None),
                         "csa2_block_size": args.csa2_block_size if args.csa2 else None},
        "store_tokens": int(original_tokens), "compact_tokens": int(compact_tokens),
        "round_trip": bool(matches),
        "parameters": int(parameters), "liger": swapped,
        "batch": args.batch, "length": args.length, "steps": steps,
        "accumulate": args.accumulate,
        "scored_tokens": steps * window,
        "seconds": elapsed, "tokens_per_second": steps * window / elapsed,
        "first_loss": history[0]["loss"], "final_loss": history[-1]["loss"],
        "inflation": inflation,
        "final_loss_per_original_token": history[-1]["loss_per_original_token"],
        "final_heldout": history[-1].get("heldout"),
        "final_heldout_per_original_token": history[-1].get(
            "heldout_per_original_token"),
        "final_copy": history[-1].get("copy"),
        "final_routing": history[-1].get("routing"),
        "passes": history[-1]["passes"],
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30,
        "shared_delta_gib": spilled, "spilled": bool(spilled > 0.25),
        "setup_seconds": train_started - started,
        "history": history,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.no_checkpoint:
        # Same convention as copy_train: weights, tokenizer and a milestone that
        # records what produced them, so the checkpoint can answer questions the
        # report did not think to ask.
        target = args.checkpoints / ("smoke-%s" % variant)
        if target.exists() and any(target.iterdir()):
            # The identity above should make this unreachable; if it is reached, two runs
            # differ in something the name does not carry, and overwriting would destroy
            # the earlier one's weights rather than its report.
            raise SystemExit(
                "%s already holds a checkpoint; move it aside or pass --checkpoints, "
                "rather than overwriting an arm that is not this one" % target)
        target.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(target, safe_serialization=True)
        tokenizer.save_pretrained(target)
        (target / "milestone.json").write_text(json.dumps({
            "variant": variant, "seed": args.seed,
            "architecture": report["architecture"],
            "scored_tokens": steps * window, "optimizer_steps": steps,
            "final_loss": report["final_loss"],
            "final_heldout": report["final_heldout"],
            "final_copy": report["final_copy"],
            "final_routing": report["final_routing"],
            "tokens_per_second": report["tokens_per_second"],
            "peak_reserved_gib": report["peak_reserved_gib"],
        }, indent=2), encoding="utf-8")
        print("CHECKPOINT %s: %d tokens, %d steps, loss %.4f"
              % (target.name, steps * window, steps, report["final_loss"]),
              flush=True)
    print("\nloss %.4f -> %.4f over %d tokens at %.0f tok/s, peak %.2f GiB, spill %+.2f GiB"
          % (report["first_loss"], report["final_loss"], report["scored_tokens"],
             report["tokens_per_second"], report["peak_reserved_gib"], spilled),
          flush=True)
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
