"""Two cards against one, on the step this project actually trains.

The prior two-card work measured a 10-layer toy at hidden 512 and concluded "split when
the micro-batch is large; do not split a small one" -- 5.3% at micro-batch 32, a loss at
micro-batch 8. That is the right shape of answer and the wrong model: this one is 1.9B
parameters with a 248,320-wide vocabulary, runs MLA and CSA2 rather than stock attention,
and is capped at micro-batch 2 by memory rather than by anything about its arithmetic.

So the question here is not whether splitting makes a step faster. It is whether splitting
makes a *bigger* step fit, because at micro-batch 1 the card is at 1,700 tokens per second
and at micro-batch 2 it is at 2,156, and micro-batch 4 does not fit at all. The comparison
that matters is therefore the best that fits on one card against the best that fits on two,
not batch 2 against batch 2.

Memory is the reason to expect anything. Of 1,915M parameters, 906M are MLP, 379M are the
linear-attention layers and 90M are the MLA/CSA2 projections -- 72% shardable at 6 bytes
each (bf16 weight, bf16 gradient, two 8-bit optimizer states), so about 4 GiB should leave
the home card. The embedding stays whole: Cut Cross-Entropy never forms the logits, which
is the only reason a 248,320-wide vocabulary is affordable, and it needs the whole head.

Run each arm in its own process. A previous arm's allocator state is not something to
measure through.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import triton_shim  # noqa: F401,E402
import torch  # noqa: E402
import bitsandbytes as bnb  # noqa: E402
from cut_cross_entropy import linear_cross_entropy  # noqa: E402

from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402
from distillkit.models.qwen35.csa2 import (isolated_indexer,  # noqa: E402
                                           recorded_attention)
from indexer_kl import indexer_loss, routing_layers, watch  # noqa: E402

CHECKPOINT = "scratch/dense_gr/checkpoints-2b/warmed-scaled"


def warm_autotune(model, home, length, micro):
    """One single-threaded step, so Triton's autotuner never races itself.

    `Autotuner` keeps what it is benchmarking on the instance and clears it when `run`
    returns; autograd gives each device its own backward thread, so a cold autotune under
    sharding has one thread clearing `nargs` while the other is still inside `benchmark()`
    and reads None. It bites only on a cold autotune, which is why it looks intermittent.
    """
    tokens = torch.randint(0, model.config.vocab_size, (micro, length), device=home)
    torch.autograd.set_multithreading_enabled(False)
    try:
        hidden = model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                             use_cache=False).last_hidden_state
        cross_entropy(model, hidden, tokens).backward()
    finally:
        torch.autograd.set_multithreading_enabled(True)
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()


def cross_entropy(model, hidden, tokens):
    """CCE wherever the head is, with the scalar brought home.

    The head may have been moved off home to even the cards up, and callers hand its
    weight to CCE directly, so the hidden state and the labels follow it. Both moves are
    autograd nodes, so the gradient finds its way back without help.
    """
    from distillkit.parallel.latent_attention import head_device

    where = head_device(model)
    loss = linear_cross_entropy(hidden.to(where), model.lm_head.weight,
                                tokens.to(where), shift=1, reduction="mean")
    return loss.to(hidden.device)


def step(model, stage, tokens):
    """The sparse stage's step: one forward serving the language model and the indexer."""
    seen, handles = watch(model, stage)
    with recorded_attention(model), isolated_indexer(model):
        hidden = model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                             use_cache=False).last_hidden_state
        targets = {i: a.last_attention for i, a in stage}
        chosen = {i: a.last_allowed for i, a in stage}
        borrowed = {i: a.bus.require_latent(a.latent_donor, i) for i, a in stage}
        for handle in handles:
            handle.remove()
        loss = cross_entropy(model, hidden, tokens)
        loss = loss + indexer_loss(model, stage, seen, targets, borrowed, selected=chosen)
    return loss


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=int, default=1, choices=(1, 2))
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--embedding-on", type=int, default=None, metavar="CARD",
                        help="move the whole embedding and head to this card instead of "
                             "leaving them home. They cannot be split -- CCE needs the "
                             "head whole -- but they are 3.05 GiB and the cards are not "
                             "equally full.")
    parser.add_argument("--fraction", type=float, default=0.9,
                        help="share of each card the allocator may use. 0.9 of 24 GiB is "
                             "21.6, and micro-batch 8 misses that by 32 MiB. Raising it "
                             "spends the margin that keeps Windows from serving VRAM out "
                             "of system RAM over PCIe, which does not fail -- it silently "
                             "measures the bus instead of the model.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    devices = ["cuda:0"] if args.cards == 1 else ["cuda:0", "cuda:1"]
    for device in devices:
        torch.cuda.set_per_process_memory_fraction(args.fraction, torch.device(device))
    model = Qwen35WidenedForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16).to(devices[0]).train()
    total = sum(p.numel() for p in model.parameters())

    if args.cards == 2:
        from distillkit.parallel.model import shard_model

        # The head stays whole: CCE never forms the logits and needs it that way. Whole
        # is not the same as home, though, and --embedding-on moves it.
        shard_model(model, devices, shard_embeddings=False,
                    embedding_device=(None if args.embedding_on is None
                                      else "cuda:%d" % args.embedding_on))
    stage = routing_layers(model)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-8)

    resident = {}
    for name, parameter in model.named_parameters():
        resident[str(parameter.device)] = resident.get(str(parameter.device), 0) + parameter.numel()
    print("cards %d, %d routing layers, %.0fM parameters" % (args.cards, len(stage), total / 1e6))
    print("resident: %s" % {k: "%.0fM" % (v / 1e6) for k, v in sorted(resident.items())})
    print()
    print("%-8s %12s %12s %14s %12s" % ("batch", "tokens/s", "ms/step", "peak GiB 0/1", "status"))
    print("-" * 64)

    rows = []
    for batch in args.batches:
        tokens = torch.randint(1, model.config.vocab_size, (batch, args.length),
                               device=devices[0])
        # Once per batch size, not once per run: the warm-up locks in a config for the
        # shapes it saw, and a new micro-batch is a new shape, so the autotuner goes cold
        # again and the two backward threads have something fresh to race over.
        try:
            warm_autotune(model, devices[0], args.length, batch)
        except torch.OutOfMemoryError:
            pass
        for device in devices:
            torch.cuda.reset_peak_memory_stats(torch.device(device))
        torch.cuda.empty_cache()
        failed = None
        try:
            for index in range(args.steps + 2):
                if index == 2:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                step(model, stage, tokens).backward()
                optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
        except torch.OutOfMemoryError:
            failed = "OOM"
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        peaks = [torch.cuda.max_memory_allocated(torch.device(d)) / 2 ** 30
                 for d in devices]
        shown = "/".join("%.2f" % p for p in peaks)
        if failed:
            print("%-8d %12s %12s %14s %12s" % (batch, "-", "-", shown, failed))
            rows.append({"batch": batch, "status": "OOM", "peak_gib": peaks})
            break
        per_step = elapsed / args.steps
        rate = batch * args.length / per_step
        print("%-8d %12.0f %12.1f %14s %12s" % (batch, rate, per_step * 1000, shown, "ok"))
        rows.append({"batch": batch, "tokens_per_second": rate,
                     "ms_per_step": per_step * 1000, "peak_gib": peaks, "status": "ok"})

    if args.output:
        args.output.write_text(json.dumps(
            {"cards": args.cards, "length": args.length, "checkpoint": args.checkpoint,
             "parameters": total, "resident": resident, "rows": rows}, indent=2),
            encoding="utf-8")
        print("\nwrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
