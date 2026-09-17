"""Where a training step's GPU time actually goes.

Every optimization considered so far was chosen by guessing which part was expensive and
measuring end to end. This is the other direction: profile one configuration and read off
what is left, so the next thing worked on is the next thing that costs.

    CUDA_VISIBLE_DEVICES=0 python scratch/dense_gr/profile_step.py
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import sys
from collections import defaultdict
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

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from benchmark import build  # noqa: E402
from cut_cross_entropy import linear_cross_entropy  # noqa: E402
from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402

#: Kernels grouped by what part of the model they belong to. Order matters: the first
#: pattern that matches a kernel name wins.
GROUPS = [
    ("gr routing", ("branchnorm", "gatedmean", "_branch", "rsqrt", "sigmoid", "silu")),
    ("loss (cce)", ("cce", "linear_cross_entropy", "_cce")),
    ("attention", ("flash", "attn", "softmax", "bmm")),
    ("linear attention", ("delta_rule", "chunk_", "conv1d", "gated_delta")),
    ("matmul", ("gemm", "cutlass", "sgemm", "ampere", "nn_", "tn_", "nt_")),
    ("optimizer", ("adam", "optimizer", "clip", "foreach", "norm_")),
    ("elementwise", ("elementwise", "vectorized", "copy", "cast", "fill", "add", "mul")),
]


def classify(name):
    lowered = name.lower()
    for group, patterns in GROUPS:
        if any(pattern in lowered for pattern in patterns):
            return group
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--vocab", type=int, default=32_768)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--branches", type=int, default=4,
                        help="1 isolates what the four-stream route costs")
    args = parser.parse_args()

    torch.cuda.set_per_process_memory_fraction(0.90, 0)
    config = build(args.hidden, args.layers, args.vocab, branches=args.branches,
                   attn_implementation="flash_attention_2")
    torch.manual_seed(0)
    model = Qwen35WidenedForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    if args.checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
    tokens = torch.randint(0, args.vocab, (args.batch, args.length), device="cuda")
    attention = torch.ones_like(tokens)

    def step():
        optimizer.zero_grad(set_to_none=True)
        hidden = model.model(input_ids=tokens, attention_mask=attention,
                             use_cache=False).last_hidden_state
        loss = linear_cross_entropy(hidden, model.lm_head.weight, tokens, shift=1,
                                    reduction="mean")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    for _ in range(3):
        step()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(args.steps):
            step()
        torch.cuda.synchronize()

    totals = defaultdict(float)
    counts = defaultdict(int)
    kernels = defaultdict(float)
    for event in prof.key_averages():
        micros = getattr(event, "self_device_time_total", 0) or 0
        if micros <= 0:
            continue
        group = classify(event.key)
        totals[group] += micros
        counts[group] += event.count
        kernels[event.key] += micros

    grand = sum(totals.values()) or 1.0
    print("%-20s %10s %8s %10s" % ("group", "ms/step", "share", "launches"))
    for group, micros in sorted(totals.items(), key=lambda kv: -kv[1]):
        print("%-20s %10.2f %7.1f%% %10d"
              % (group, micros / 1000 / args.steps, 100 * micros / grand,
                 counts[group] // args.steps))
    print("%-20s %10.2f" % ("TOTAL", grand / 1000 / args.steps))

    print("\ntop kernels by device time:")
    for name, micros in sorted(kernels.items(), key=lambda kv: -kv[1])[:14]:
        print("  %7.2f ms  %5.1f%%  %s"
              % (micros / 1000 / args.steps, 100 * micros / grand, name[:82]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
