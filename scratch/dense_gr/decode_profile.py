"""Where a decode step's 100 ms goes.

A 1.9B model in bfloat16 should generate a token in single-digit milliseconds. This one
takes about 100, and the figure barely moves between a 1K and a 16K cache -- so whatever
dominates is not attention, not the KV cache, and not anything this project has been
optimizing. That makes every cache ratio measured so far a statement about memory and not
about tokens per second.

Three questions, in the order that narrows fastest:

    is the GPU even busy   kernel time against wall time. A step that is mostly idle is
                           launch-bound, and the fix is fewer, bigger kernels rather than
                           less arithmetic.
    which layers           the stack is 18 recurrent layers and 6 attention ones, and
                           they are entirely different code paths.
    which operators        the top of the profiler's own table, by self CUDA time and by
                           call count -- a thousand tiny launches and one slow kernel
                           look the same from above and want opposite fixes.
"""
from __future__ import annotations

import argparse
import collections
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


def per_layer_times(model, step, cache, repeats):
    """Wall time inside each layer, by the kind of layer it is."""
    spent = collections.defaultdict(float)
    kinds = {i: ("full" if "linear" not in str(k) else "linear")
             for i, k in enumerate(model.config.layer_types)}
    marks = {}

    def before(index):
        def hook(module, args, kwargs):
            torch.cuda.synchronize()
            marks[index] = time.perf_counter()
        return hook

    def after(index):
        def hook(module, args, kwargs, output):
            torch.cuda.synchronize()
            spent[kinds[index]] += time.perf_counter() - marks[index]
        return hook

    handles = []
    for index, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_pre_hook(before(index), with_kwargs=True))
        handles.append(layer.register_forward_hook(after(index), with_kwargs=True))
    with torch.no_grad():
        for _ in range(repeats):
            model(input_ids=step, past_key_values=cache, use_cache=True)
    for handle in handles:
        handle.remove()
    return {k: v * 1000.0 / repeats for k, v in spent.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    stream = open_split(args.store, "heldout", 248320)
    ids = torch.from_numpy(
        np.array(stream[:args.context + 1], dtype=np.int64).reshape(1, -1))
    model = Qwen35WidenedForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to(device).eval()
    model.config.use_cache = True

    with torch.no_grad():
        out = model(input_ids=ids[:, :args.context].to(device), use_cache=True)
    cache = out.past_key_values
    step = ids[:, -1:].to(device)

    with torch.no_grad():
        model(input_ids=step, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()
        began = time.perf_counter()
        for _ in range(args.repeats):
            model(input_ids=step, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()
    wall = (time.perf_counter() - began) * 1000.0 / args.repeats
    print("context %d, %.2f ms per decode step\n" % (args.context, wall))

    layers = per_layer_times(model, step, cache, args.repeats)
    counts = collections.Counter(
        "full" if "linear" not in str(k) else "linear"
        for k in model.config.layer_types)
    print("%-10s %6s %10s %10s" % ("layer kind", "count", "total ms", "each ms"))
    print("-" * 40)
    for kind, total in sorted(layers.items(), key=lambda kv: -kv[1]):
        print("%-10s %6d %10.2f %10.3f"
              % (kind, counts[kind], total, total / max(counts[kind], 1)))
    print("%-10s %6s %10.2f  (the rest is embedding, norm and the head)"
          % ("outside", "", wall - sum(layers.values())))

    from torch.profiler import ProfilerActivity, profile

    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as traced:
            for _ in range(args.repeats):
                model(input_ids=step, past_key_values=cache, use_cache=True)
            torch.cuda.synchronize()

    events = traced.key_averages()
    kernel = sum(e.self_device_time_total for e in events) / 1000.0 / args.repeats
    host = sum(e.self_cpu_time_total for e in events) / 1000.0 / args.repeats
    launches = sum(e.count for e in events) / args.repeats
    print("\n%.0f operators dispatched per token, %.2f ms of host time, %.2f ms wall"
          % (launches, host, wall))
    if kernel <= 0.0:
        # CUPTI does not always initialize on this box, and when it does not the device
        # column is zeros rather than missing. Saying "the GPU was idle" off a counter
        # that never started is the kind of claim this file exists to avoid.
        print("Device timing unavailable (CUPTI did not initialize); host time and the")
        print("dispatch count are measured, the GPU's share is not.")
    else:
        print("GPU busy %.2f ms, %.0f%% of wall" % (kernel, 100.0 * kernel / wall))

    print("\n%-40s %9s %9s" % ("operator, by count", "calls", "host ms"))
    print("-" * 62)
    for event in sorted(events, key=lambda e: -e.count)[:14]:
        print("%-40s %9.0f %9.3f"
              % (event.key[:40], event.count / args.repeats,
                 event.self_cpu_time_total / 1000.0 / args.repeats))

    print("\n%-40s %9s %9s" % ("operator, by host time", "calls", "host ms"))
    print("-" * 62)
    for event in sorted(events, key=lambda e: -e.self_cpu_time_total)[:14]:
        print("%-40s %9.0f %9.3f"
              % (event.key[:40], event.count / args.repeats,
                 event.self_cpu_time_total / 1000.0 / args.repeats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
