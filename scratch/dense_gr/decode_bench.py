"""What folding the up-projection into the query is worth, per decode step.

Without it the up-projection runs over the whole cached history every step -- a 384 to
3584 matrix multiply over every cached token, per layer, per token generated -- which is
the trade MLA makes and the reason the reference absorbs at serving time. The cache is
smaller either way; the question is whether reading it costs more than the memory it
saved.

Both paths are the same function. `q . (W_k c) = (W_k^T q) . c` and the value side folds
the same way, so this measures the arithmetic and not an approximation. The expanding path
is forced by giving the layers an identity key norm, which changes no number and sends the
forward down the other branch.

    python scratch/dense_gr/decode_bench.py \\
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
from distillkit.models.qwen35.csa2 import Qwen35SparseLatentAttention  # noqa: E402

from convert_full import STORE, open_split  # noqa: E402


def timed_decode(model, ids, prefill, steps, device):
    """Prefill, then time `steps` single-token steps through the cache."""
    with torch.no_grad():
        out = model(input_ids=ids[:, :prefill].to(device), use_cache=True)
        cache = out.past_key_values
        torch.cuda.synchronize()
        began = time.perf_counter()
        for index in range(steps):
            token = ids[:, prefill + index: prefill + index + 1].to(device)
            out = model(input_ids=token, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
        torch.cuda.synchronize()
    return (time.perf_counter() - began) * 1000.0 / steps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[1024, 4096, 16384])
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    stream = open_split(args.store, "heldout", 248320)
    longest = max(args.contexts) + args.steps + 1
    ids = torch.from_numpy(
        np.array(stream[:longest], dtype=np.int64).reshape(1, longest))

    print("%-10s %14s %14s %9s" % ("context", "absorbed", "expanded", "speedup"))
    print("-" * 52)
    for prefill in args.contexts:
        row = {}
        for label in ("absorbed", "expanded"):
            model = Qwen35WidenedForCausalLM.from_pretrained(
                args.model, dtype=torch.bfloat16).to(device).eval()
            model.config.use_cache = True
            if label == "expanded":
                for layer in model.model.layers:
                    inner = getattr(layer, "self_attn", None)
                    if isinstance(inner, Qwen35SparseLatentAttention):
                        inner.k_norm = torch.nn.Identity()
            try:
                row[label] = timed_decode(model, ids, prefill, args.steps, device)
            except Exception as error:  # noqa: BLE001 - a refusal is the measurement
                row[label] = float("nan")
                print("  %s at %d: %s"
                      % (label, prefill, ("%s" % error).splitlines()[0][:80]))
            del model
            torch.cuda.empty_cache()
        speedup = row["expanded"] / row["absorbed"] if row["absorbed"] else float("nan")
        print("%-10d %11.2f ms %11.2f ms %8.2fx"
              % (prefill, row["absorbed"], row["expanded"], speedup), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
