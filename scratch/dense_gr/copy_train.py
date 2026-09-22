"""One arm of the copy/induction displacement experiment.

`docs/standard_parts.md` section 5. Three arms -- `plain`, `copy`, `scrambled` -- trained
at matched compute on the 3B corpus, scored on held-out NLL and on the copy probe under
three conditions. The decisive cell is `copy` with the module ablated: if that collapses
toward chance while `plain` performs well, the backbone genuinely stopped building the
function.

    CUDA_VISIBLE_DEVICES=0 python scratch/dense_gr/copy_train.py --arm copy --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import triton_shim  # noqa: F401,E402  resolves triton-windows before CCE reads the version

import numpy as np  # noqa: E402
import torch  # noqa: E402

from benchmark import (ATTENTION_RATIOS, SpillWatch, apply_liger, build,  # noqa: E402
                       variant_tag)
from copy_module import ARMS, Connector, build_inputs, module_output  # noqa: E402
from copy_probe import copy_probe, format_probe  # noqa: E402
from cut_cross_entropy import linear_cross_entropy  # noqa: E402
from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402
from vocab_remap import build_vocabulary, cached_remap  # noqa: E402

BASE = "D:/DeepThought/Projects/HybridModel/student-2b-hf"
STORE = Path("scratch/code_training/tokens-v2")


def write_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def sha256_of_directory(path: Path) -> str:
    """Digest of a checkpoint's weight files, so an artifact can name what produced it."""
    hasher = hashlib.sha256()
    for file in sorted(path.glob("*.safetensors")):
        with open(file, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 22), b""):
                hasher.update(block)
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--spill-every", type=float, default=30.0,
                        help="seconds between background spill checks; the counter\n"
                             "read costs 1.8 s, so it is polled off the training thread")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab", type=int, default=32_768)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--ratio", default="1:1", choices=sorted(ATTENTION_RATIOS),
                        help="gated-delta-net layers per full-attention layer; "
                             "1:1 is what these arms have run, 3:1 is Qwen3-Next's")
    parser.add_argument("--blend", type=float, default=0.0,
                        help="gated residual route strength; 0 leaves it inert, "
                             "which is what every arm so far has run")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8_000,
                        help="the copy probe saturates by ~5,000 steps, so stage A stops "
                             "shortly after rather than running a full pass")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=8)
    parser.add_argument("--evaluate-every", type=int, default=1_000)
    parser.add_argument("--evaluate-windows", type=int, default=64)
    parser.add_argument("--probe-every", type=int, default=250)
    parser.add_argument("--probe-half", type=int, default=256)
    parser.add_argument("--probe-windows", type=int, default=64)
    parser.add_argument("--report-every", type=int, default=250)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--checkpoints", type=Path,
                        default=Path("scratch/dense_gr/checkpoints"),
                        help="root for saved arms; the baseline run saved nothing, so "
                             "asking it a question it did not already record means "
                             "retraining it")
    parser.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args()
    variant = variant_tag(args.ratio, args.blend, seed=args.seed)
    stem = "%s-s%d-%s" % (args.arm, args.seed, variant)
    if args.output is None:
        args.output = Path("scratch/dense_gr/copy-%s.json" % stem)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.cuda.set_per_process_memory_fraction(0.90, 0)
    started = time.perf_counter()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)

    raw = np.memmap(args.store / "train.bin", dtype=np.uint32, mode="r")
    counts = np.load(args.store / "train-counts.npy")
    kept, forward, bytes_ids = build_vocabulary(counts, tokenizer, args.vocab)
    stream, original_tokens, compact_tokens = cached_remap(
        args.store, "train", args.vocab, tokenizer, forward, bytes_ids, counts, kept)
    stream = np.asarray(stream)
    inflation = compact_tokens / original_tokens
    print("arm %s seed %d | store %d -> %d, inflation %.4f"
          % (args.arm, args.seed, original_tokens, compact_tokens, inflation), flush=True)

    config = build(args.hidden, args.layers, args.vocab, ratio=args.ratio,
                   blend=args.blend,
                   attn_implementation="flash_attention_2")
    torch.manual_seed(args.seed)
    model = Qwen35WidenedForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    model.train()
    swapped = apply_liger(model, config)

    connector = None
    if args.arm != "plain":
        connector = Connector(args.hidden).to(device="cuda", dtype=torch.bfloat16)
    backbone_parameters = sum(p.numel() for p in model.parameters())
    connector_parameters = 0 if connector is None else sum(
        p.numel() for p in connector.parameters())
    print("model: %.1fM backbone, %.3fM connector (%.2f%%), liger %s"
          % (backbone_parameters / 1e6, connector_parameters / 1e6,
             100 * connector_parameters / backbone_parameters, json.dumps(swapped)),
          flush=True)

    import bitsandbytes as bnb
    trainable = list(model.parameters()) + (
        [] if connector is None else list(connector.parameters()))
    optimizer = bnb.optim.AdamW8bit(trainable, lr=args.lr, betas=(0.9, 0.95),
                                    weight_decay=0.1)
    # Polled on a daemon thread: reading it inline costs 1.81 s per report and
    # leaves the GPU with nothing queued for all of it.
    spill = SpillWatch(interval=args.spill_every).start()

    module_rng = np.random.default_rng(args.seed + 9_000)

    def hidden_for(tokens, arm=None, ablate=False, substitute=False, ids=None):
        """Run the backbone on `tokens` under one treatment of the module.

        `ids` is the same batch as a host array where the caller already has one. The
        training loop does, and reading it back off the device instead would force a
        synchronize on every step.
        """
        arm = args.arm if arm is None else arm
        if ids is None:
            ids = tokens.detach().cpu().numpy()
        features, candidate = module_output(ids, arm, args.vocab, module_rng,
                                            args.order, args.max_length)
        if substitute and features is not None:
            from copy_module import scramble
            features, candidate = scramble(features, candidate, module_rng)
        if features is None:
            inputs = model.model.embed_tokens(tokens)
        else:
            inputs = build_inputs(
                model, tokens,
                torch.from_numpy(features).to("cuda"),
                torch.from_numpy(candidate).to("cuda"),
                connector, ablate=ablate)
        return model.model(inputs_embeds=inputs,
                           attention_mask=torch.ones_like(tokens),
                           use_cache=False).last_hidden_state

    held_stream, _, _ = cached_remap(args.store, "calibration", args.vocab, tokenizer,
                                     forward, bytes_ids, counts, kept)
    held_rng = np.random.default_rng(12345)
    held_starts = held_rng.integers(0, held_stream.shape[0] - args.length - 1,
                                    size=args.evaluate_windows)
    evaluation = torch.from_numpy(
        np.stack([held_stream[s:s + args.length] for s in held_starts]).astype(np.int64))

    @torch.no_grad()
    def evaluate():
        model.eval()
        total, batches = 0.0, 0
        for start in range(0, evaluation.shape[0], args.batch):
            chunk = evaluation[start:start + args.batch].to("cuda")
            state = hidden_for(chunk)
            total += float(linear_cross_entropy(state, model.lm_head.weight, chunk,
                                                shift=1, reduction="mean"))
            batches += 1
        model.train()
        return total / max(batches, 1)

    def probe(arm=None, ablate=False, substitute=False):
        return copy_probe(model, args.vocab, half=args.probe_half,
                          windows=args.probe_windows, batch=args.batch,
                          hidden_fn=lambda ids: hidden_for(ids, arm, ablate, substitute))

    window = args.batch * args.length
    generator = np.random.default_rng(args.seed)
    history = []
    torch.cuda.synchronize()
    train_started = time.perf_counter()
    for step in range(args.steps):
        starts = generator.integers(0, stream.shape[0] - args.length - 1, size=args.batch)
        batch = np.stack([stream[s:s + args.length] for s in starts]).astype(np.int64)
        tokens = torch.from_numpy(batch).to("cuda", non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        hidden = hidden_for(tokens, ids=batch)
        loss = linear_cross_entropy(hidden, model.lm_head.weight, tokens, shift=1,
                                    reduction="mean")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if spill.breached():
            # Set by the watcher thread the moment a poll exceeded the tolerance, so the
            # step this stops on is the first one after the breach rather than the next
            # multiple of report_every.
            args.output.write_text(json.dumps(
                {"aborted": "spilled to system RAM",
                 "shared_delta_gib": spill.tripped_at, "arm": args.arm, "seed": args.seed,
                 "step": step, "history": history}, indent=2), encoding="utf-8")
            raise SystemExit("spilled %.2f GiB into system RAM at step %d; stopping"
                             % (spill.tripped_at, step))

        if step % args.report_every == 0 or step == args.steps - 1:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - train_started
            seen = (step + 1) * window
            scored = float(loss.detach())
            row = {"step": step, "tokens": seen, "loss": scored,
                   "loss_per_original_token": scored * inflation,
                   "tokens_per_second": seen / elapsed}
            if step % args.evaluate_every == 0 or step == args.steps - 1:
                row["heldout"] = evaluate()
                row["heldout_per_original_token"] = row["heldout"] * inflation
            if args.probe_every and (step % args.probe_every == 0
                                     or step == args.steps - 1):
                row["copy"] = probe()
            history.append(row)
            print("step %5d  train %7.4f  held %8s  %8.0f tok/s"
                  % (step, row["loss"],
                     "%.4f" % row["heldout"] if "heldout" in row else "-",
                     row["tokens_per_second"]), flush=True)
            if "copy" in row:
                print("            %s" % format_probe(row["copy"]), flush=True)
            row.update(spill.report())

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - train_started

    # The endpoint matrix. `ablated` is the decisive cell; `substituted` separates
    # dependence on content from dependence on the channel merely being active.
    endpoint = {"enabled": probe()}
    if args.arm != "plain":
        endpoint["ablated"] = probe(ablate=True)
        endpoint["substituted"] = probe(substitute=True)
    for name, result in endpoint.items():
        print("endpoint %-12s %s" % (name, format_probe(result)), flush=True)

    spill.stop()
    report = {
        "arm": args.arm, "seed": args.seed, "vocab": args.vocab,
        "architecture": {"ratio": args.ratio, "blend": args.blend,
                         "variant": variant, "hidden": args.hidden,
                         "layers": args.layers,
                         "full_attention_layers": [
                             index for index, kind in enumerate(config.layer_types)
                             if "linear" not in str(kind)],
                         "mla": bool(getattr(config, "mla_enabled", False)),
                         "csa2": bool(getattr(config, "csa2_enabled", False))},
        "backbone_parameters": int(backbone_parameters),
        "connector_parameters": int(connector_parameters),
        "batch": args.batch, "length": args.length, "steps": args.steps,
        "scored_tokens": args.steps * window, "inflation": inflation,
        "order": args.order, "max_length": args.max_length,
        "seconds": elapsed, "tokens_per_second": args.steps * window / elapsed,
        "final_loss": history[-1]["loss"],
        "final_heldout": history[-1].get("heldout"),
        "final_heldout_per_original_token": history[-1].get(
            "heldout_per_original_token"),
        "endpoint": endpoint,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30,
        **spill.report(),
        "setup_seconds": train_started - started,
        "history": history,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote %s" % args.output)

    if not args.no_checkpoint:
        target = args.checkpoints / stem
        target.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(target, safe_serialization=True)
        tokenizer.save_pretrained(target)
        if connector is not None:
            # The connector is not part of the backbone and `save_pretrained` will not see
            # it. Saved beside it with its own provenance, the way `code_gate/train_gate.py`
            # stores a gate: an arm is only reproducible if the piece bolted on comes back
            # with it.
            torch.save({"state_dict": {k: v.cpu() for k, v in
                                       connector.state_dict().items()},
                        "arm": args.arm, "seed": args.seed, "hidden": args.hidden,
                        "order": args.order, "max_length": args.max_length,
                        "vocab": args.vocab, "steps": args.steps},
                       target / "connector.pt")
        write_atomic(target / "milestone.json", {
            "arm": args.arm, "seed": args.seed,
            "scored_tokens": args.steps * window, "optimizer_steps": args.steps,
            "final_loss": report["final_loss"], "final_heldout": report["final_heldout"],
            "endpoint": endpoint,
            "elapsed_seconds": elapsed, "tokens_per_second": report["tokens_per_second"],
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "sha256": sha256_of_directory(target),
        })
        print("CHECKPOINT %s: %d tokens, %d steps, loss %.4f"
              % (target.name, args.steps * window, args.steps, report["final_loss"]),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
