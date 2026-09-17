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

from benchmark import apply_liger, build, shared_gpu_gib  # noqa: E402
from cut_cross_entropy import linear_cross_entropy  # noqa: E402
from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402

BASE = "D:/DeepThought/Projects/HybridModel/student-2b-hf"
STORE = Path("scratch/code_training/tokens")


def bytes_to_unicode():
    """The byte-level BPE alphabet: each of the 256 bytes as a printable character.

    Defined here rather than imported. It is a fixed mapping that every byte-level BPE
    tokenizer shares, and transformers has moved it twice -- it is currently inside
    `convert_slow_tokenizer`, which is not an interface to depend on.
    """
    printable = (list(range(ord("!"), ord("~") + 1))
                 + list(range(ord("\xa1"), ord("\xac") + 1))
                 + list(range(ord("\xae"), ord("\xff") + 1)))
    mapped = printable[:]
    spare = 0
    for value in range(256):
        if value not in printable:
            printable.append(value)
            mapped.append(256 + spare)
            spare += 1
    return dict(zip(printable, (chr(point) for point in mapped)))


def byte_token_ids(tokenizer):
    """The id of each of the 256 single-byte tokens, in byte order.

    This is byte-level BPE, so every byte has a token. The alphabet above is the map the
    tokenizer itself uses to make bytes printable; inverting it turns a token string back
    into the bytes it stands for.
    """
    encoder = bytes_to_unicode()
    vocabulary = tokenizer.get_vocab()
    ids = []
    for value in range(256):
        token = encoder[value]
        if token not in vocabulary:
            raise SystemExit("byte %d has no token; byte fallback would have holes" % value)
        ids.append(vocabulary[token])
    return ids


def build_vocabulary(counts, tokenizer, target):
    """Keep the bytes, the specials, then the most frequent ids until `target` is full."""
    specials = sorted(tokenizer.get_added_vocab().values())
    bytes_ids = byte_token_ids(tokenizer)
    forced = list(dict.fromkeys(bytes_ids + specials))
    if len(forced) > target:
        raise SystemExit("target %d cannot hold %d forced ids" % (target, len(forced)))

    order = np.argsort(counts)[::-1]
    kept = list(forced)
    seen = set(forced)
    for candidate in order:
        if len(kept) >= target:
            break
        candidate = int(candidate)
        if candidate not in seen:
            kept.append(candidate)
            seen.add(candidate)
    kept.sort()
    forward = np.full(counts.shape[0], -1, dtype=np.int32)
    forward[np.asarray(kept)] = np.arange(len(kept), dtype=np.int32)
    return kept, forward, bytes_ids


def expansion_for(original, tokenizer, forward, bytes_ids):
    """The compact ids an unkept token becomes: its own bytes."""
    decoder = {char: value for value, char in bytes_to_unicode().items()}
    token = tokenizer.convert_ids_to_tokens(int(original))
    pieces = []
    for char in token:
        if char not in decoder:
            return None
        pieces.append(int(forward[bytes_ids[decoder[char]]]))
    return pieces


def remap(tokens, tokenizer, forward, bytes_ids, kept):
    """Original ids to compact ids, expanding anything below the cut into its bytes.

    `uint16` only while the compact space fits in it, which is exactly up to 65,536 --
    the largest cut that halves the store's width. Above that the store stays `uint32`
    and the only saving is the embedding.
    """
    width = np.uint16 if len(kept) <= 65_536 else np.uint32
    compact = forward[tokens]
    missing = np.flatnonzero(compact < 0)
    if missing.size == 0:
        return compact.astype(width), 0
    cache = {}
    pieces, last = [], 0
    for index in missing:
        original = int(tokens[index])
        if original not in cache:
            expanded = expansion_for(original, tokenizer, forward, bytes_ids)
            if expanded is None:
                raise SystemExit("token %d could not be decomposed into bytes" % original)
            cache[original] = np.asarray(expanded, dtype=np.int32)
        pieces.append(compact[last:index])
        pieces.append(cache[original])
        last = index + 1
    pieces.append(compact[last:])
    return np.concatenate(pieces).astype(width), int(missing.size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=int, default=16_384)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--length", type=int, default=1024)
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
    parser.add_argument("--output", type=Path,
                        default=Path("scratch/dense_gr/smoke-train.json"))
    args = parser.parse_args()

    torch.cuda.set_per_process_memory_fraction(0.90, 0)
    started = time.perf_counter()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)

    raw = np.memmap(STORE / "train.bin", dtype=np.uint32, mode="r")
    print("store: %d tokens" % raw.shape[0], flush=True)
    counts = np.bincount(np.asarray(raw, dtype=np.int64), minlength=248_320)
    kept, forward, bytes_ids = build_vocabulary(counts, tokenizer, args.vocab)
    coverage = float(counts[np.asarray(kept)].sum() / counts.sum())
    print("vocabulary: kept %d ids, %.4f%% coverage" % (len(kept), 100 * coverage),
          flush=True)

    stream, expanded = remap(np.asarray(raw), tokenizer, forward, bytes_ids, kept)
    print("remap: %d tokens -> %d (%d expanded to bytes, %.4f%%)"
          % (raw.shape[0], stream.shape[0], expanded,
             100 * expanded / raw.shape[0]), flush=True)

    # Round trip: the compact stream must decode to what the original ids decode to.
    inverse = np.asarray(kept, dtype=np.int64)
    sample = stream[:4096].astype(np.int64)
    round_tripped = tokenizer.decode(inverse[sample].tolist())
    reference = tokenizer.decode(np.asarray(raw[:4096]).tolist())
    matches = round_tripped[:2000] == reference[:2000]
    print("round trip on the first 4096 compact tokens: %s" % matches, flush=True)

    config = build(args.hidden, args.layers, args.vocab,
                   attn_implementation="flash_attention_2")
    torch.manual_seed(0)
    model = Qwen35WidenedForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    model.train()
    swapped = apply_liger(model, config)
    parameters = sum(p.numel() for p in model.parameters())
    print("model: %.1fM parameters, liger %s" % (parameters / 1e6, json.dumps(swapped)),
          flush=True)

    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                    weight_decay=0.1)
    baseline_shared = shared_gpu_gib()

    # Held-out: the calibration split shares no repository with train, so this measures
    # generalization rather than how much of the corpus has been memorized -- which is
    # what training loss becomes once a budget spans several passes.
    evaluation = None
    if args.evaluate_every:
        held = np.memmap(STORE / "calibration.bin", dtype=np.uint32, mode="r")
        held_stream, held_expanded = remap(np.asarray(held), tokenizer, forward,
                                           bytes_ids, kept)
        rng = np.random.default_rng(12345)
        starts = rng.integers(0, held_stream.shape[0] - args.length - 1,
                              size=args.evaluate_windows)
        evaluation = torch.from_numpy(
            np.stack([held_stream[s:s + args.length] for s in starts]).astype(np.int64))
        print("held-out: %d tokens -> %d, %d fixed windows"
              % (held.shape[0], held_stream.shape[0], args.evaluate_windows), flush=True)

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
    inflation = stream.shape[0] / raw.shape[0]

    window = args.batch * args.length
    if args.passes is not None:
        args.tokens = int(args.passes * stream.shape[0])
    steps = max(1, args.tokens // window)
    generator = np.random.default_rng(0)
    history = []
    torch.cuda.synchronize()
    train_started = time.perf_counter()
    for step in range(steps):
        starts = generator.integers(0, stream.shape[0] - args.length - 1,
                                    size=args.batch)
        batch = np.stack([stream[s:s + args.length] for s in starts]).astype(np.int64)
        tokens = torch.from_numpy(batch).to("cuda", non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        hidden = model.model(input_ids=tokens,
                             attention_mask=torch.ones_like(tokens),
                             use_cache=False).last_hidden_state
        loss = linear_cross_entropy(hidden, model.lm_head.weight, tokens, shift=1,
                                    reduction="mean")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.report_every == 0 or step == steps - 1:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - train_started
            seen = (step + 1) * window
            row = {"step": step, "tokens": seen, "passes": seen / stream.shape[0],
                   "loss": float(loss), "loss_per_original_token": float(loss) * inflation,
                   "tokens_per_second": seen / elapsed}
            if evaluation is not None and (step % args.evaluate_every == 0
                                           or step == steps - 1):
                row["heldout"] = evaluate()
                row["heldout_per_original_token"] = row["heldout"] * inflation
            history.append(row)
            print("step %5d  %5.2f passes  train %7.4f  held %8s  norm %7.4f  %8.0f tok/s"
                  % (step, row["passes"], row["loss"],
                     "%.4f" % row["heldout"] if "heldout" in row else "-",
                     row.get("heldout_per_original_token",
                             row["loss_per_original_token"]),
                     row["tokens_per_second"]), flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - train_started
    spilled = shared_gpu_gib() - baseline_shared
    report = {
        "vocab": args.vocab, "kept_ids": len(kept), "coverage": coverage,
        "store_tokens": int(raw.shape[0]), "compact_tokens": int(stream.shape[0]),
        "expanded_tokens": expanded, "round_trip": bool(matches),
        "parameters": int(parameters), "liger": swapped,
        "batch": args.batch, "length": args.length, "steps": steps,
        "scored_tokens": steps * window,
        "seconds": elapsed, "tokens_per_second": steps * window / elapsed,
        "first_loss": history[0]["loss"], "final_loss": history[-1]["loss"],
        "inflation": inflation,
        "final_loss_per_original_token": history[-1]["loss_per_original_token"],
        "final_heldout": history[-1].get("heldout"),
        "final_heldout_per_original_token": history[-1].get(
            "heldout_per_original_token"),
        "passes": history[-1]["passes"],
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30,
        "shared_delta_gib": spilled, "spilled": bool(spilled > 0.25),
        "setup_seconds": train_started - started,
        "history": history,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nloss %.4f -> %.4f over %d tokens at %.0f tok/s, peak %.2f GiB, spill %+.2f GiB"
          % (report["first_loss"], report["final_loss"], report["scored_tokens"],
             report["tokens_per_second"], report["peak_reserved_gib"], spilled),
          flush=True)
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
