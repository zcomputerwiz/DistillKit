"""What a decode step costs once the dispatches are collapsed into a graph.

Profiling put 95 ms of host time against 90 ms of wall for a single token, across roughly
eleven thousand operator dispatches -- the GPU waiting on Python, not the other way round.
Nothing this project has optimized touches that: the cache is smaller, the attention is
sparse, and the step costs what it costs because of how many times it calls into the
driver. The unconverted source measures the same, so it is inherited rather than
introduced.

Decode is the textbook case for fixing this. The shapes are static once the cache stops
growing -- one token in, a preallocated buffer written at a moving offset -- so the step
can be captured once and replayed. `StaticCache` preallocates and `torch.compile` with
`reduce-overhead` puts it behind CUDA graphs.

Both models are measured the same way, because a speedup that only appears on the
converted one would say something very different from a speedup that appears on both.

    python scratch/dense_gr/decode_compiled.py \\
        --model scratch/dense_gr/checkpoints-conv/student-2b-csa2-allfull
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import triton_shim  # noqa: F401,E402

from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402

from convert_full import STORE, open_split  # noqa: E402


def build(path, device):
    model = Qwen35WidenedForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16).to(device).eval()
    model.config.use_cache = True
    return model


def run(model, ids, prefill, steps, device, compiled, maximum):
    """Prefill into a static cache, then time single-token steps."""
    from transformers import StaticCache

    cache = StaticCache(config=model.config, max_cache_len=maximum,
                        max_batch_size=1, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        model(input_ids=ids[:, :prefill].to(device), past_key_values=cache,
              use_cache=True,
              cache_position=torch.arange(prefill, device=device))

    step = model
    if compiled:
        step = torch.compile(model, mode="reduce-overhead", fullgraph=False)

    # One static input buffer, written in place: a fresh tensor every step would move the
    # address a captured graph was recorded against.
    token = torch.zeros(1, 1, dtype=torch.long, device=device)
    position = torch.zeros(1, dtype=torch.long, device=device)

    def once(index):
        token.copy_(ids[:, prefill + index: prefill + index + 1])
        position.fill_(prefill + index)
        with torch.no_grad():
            return step(input_ids=token, past_key_values=cache, use_cache=True,
                        cache_position=position)

    # Warm up: compilation, then the first couple of graph replays.
    for index in range(4):
        once(index)
    torch.cuda.synchronize()
    began = time.perf_counter()
    for index in range(steps):
        once(4 + index)
    torch.cuda.synchronize()
    return (time.perf_counter() - began) * 1000.0 / steps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=None,
                        help="a second model measured the same way, to tell an inherited "
                             "cost from an introduced one")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    stream = open_split(args.store, "heldout", 248320)
    span = args.context + args.steps + 8
    ids = torch.from_numpy(np.array(stream[:span], dtype=np.int64).reshape(1, span))

    paths = [("converted", args.model)]
    if args.source is not None:
        paths.append(("source", args.source))

    print("%-12s %13s %13s %9s" % ("model", "eager", "compiled", "speedup"))
    print("-" * 52)
    for label, path in paths:
        timings = {}
        for mode in (False, True):
            model = build(path, device)
            try:
                timings[mode] = run(model, ids, args.context, args.steps, device, mode,
                                    span)
            except Exception as error:  # noqa: BLE001 - a refusal is the measurement
                timings[mode] = float("nan")
                print("  %s %s: %s" % (label, "compiled" if mode else "eager",
                                       ("%s" % error).splitlines()[0][:90]))
            del model
            torch.cuda.empty_cache()
        ratio = timings[False] / timings[True] if timings[True] else float("nan")
        print("%-12s %10.2f ms %10.2f ms %8.2fx"
              % (label, timings[False], timings[True], ratio), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
