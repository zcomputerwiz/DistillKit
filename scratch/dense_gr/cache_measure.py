"""Does the converted model decode, and is its cache the size the arithmetic claims.

Every cache ratio in PROGRESS.md comes from `cached_numbers_per_token`, whose own
docstring calls itself aspirational: it reports what the latent form *would* cost. That is
the number the whole VRAM argument rests on, and nothing has ever read it off a cache that
exists.

MLA does write one -- it stores the latent and the rotary slice and rebuilds the per-head
keys and values through `kv_b_proj` -- so an MLA-only conversion can be asked directly:

    decode      do cached steps agree with one forward over the same tokens
    size        what the cache object actually holds, per token per layer
    ratio       against the source's own cache on the same tokens

A disagreement in the first is the interesting outcome. A cache that is the right size and
the wrong contents would show up nowhere else: the conversion is evaluated with
`use_cache=False` everywhere, so no measurement so far has exercised this path at all.

    python scratch/dense_gr/cache_measure.py \\
        --source D:/DeepThought/Projects/HybridModel/student-2b-hf \\
        --converted scratch/dense_gr/checkpoints-conv/student-2b-mla-fixed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import triton_shim  # noqa: F401,E402

from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402

from convert_full import STORE, open_split  # noqa: E402


def cache_bytes(cache):
    """Every tensor the cache object is holding, counted once."""
    seen, total = set(), 0
    stack = [cache]
    while stack:
        item = stack.pop()
        if torch.is_tensor(item):
            if item.data_ptr() not in seen:
                seen.add(item.data_ptr())
                total += item.numel() * item.element_size()
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif hasattr(item, "__dict__"):
            stack.extend(vars(item).values())
    return total


def decode(model, ids, device, steps):
    """Prefill, then take `steps` single-token steps through the cache.

    The model builds the cache itself. A hybrid stack needs one that knows which layers
    are linear -- those carry recurrent state rather than keys and values -- so it is
    constructed from the config, and constructing a bare one here gets the layer types
    wrong.
    """
    # Positive indices throughout: the last step's negative slice ends at 0, which python
    # reads as the start of the sequence and hands back an empty tensor.
    first = ids.shape[1] - steps
    prompt = ids[:, :first].to(device)
    with torch.no_grad():
        out = model(input_ids=prompt, use_cache=True)
        cache = out.past_key_values
        picked = [out.logits[:, -1].float().cpu()]
        for step in range(steps):
            token = ids[:, first + step: first + step + 1].to(device)
            out = model(input_ids=token, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            picked.append(out.logits[:, -1].float().cpu())
    return torch.stack(picked), cache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    stream = open_split(args.store, "heldout", 248320)
    ids = torch.from_numpy(
        np.array(stream[:args.length], dtype=np.int64).reshape(1, args.length))

    # Two lengths, because the cache holds two different things. The 18 linear-attention
    # layers carry a recurrent state that is the same size whatever the context, and only
    # the 6 attention layers grow per token. The slope between two lengths separates them
    # without having to know either one's internals.
    short, long = args.length, args.length * 2
    rows = []
    for tag, path in (("source", args.source), ("converted", args.converted)):
        model = Qwen35WidenedForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16).to(device).eval()
        model.config.use_cache = True
        attention = [i for i, k in enumerate(model.config.layer_types)
                     if "linear" not in str(k)]
        held, gap = {}, None
        try:
            for length in (short, long):
                window = torch.from_numpy(
                    np.array(stream[:length], dtype=np.int64).reshape(1, length))
                cached, cache = decode(model, window, device, args.steps)
                held[length] = cache_bytes(cache)
                if gap is None:
                    with torch.no_grad():
                        whole = model(input_ids=window.to(device),
                                      use_cache=False).logits
                    reference = whole[0, -args.steps - 1:].float().cpu()
                    gap = float((cached.squeeze(1) - reference).abs().max())
                del cache
                torch.cuda.empty_cache()
        except Exception as error:  # noqa: BLE001 - a refusal is the measurement
            print("%-10s cannot decode: %s"
                  % (tag, ("%s" % error).strip().splitlines()[0][:96]))
            del model
            torch.cuda.empty_cache()
            continue
        per_token = (held[long] - held[short]) / (long - short)
        fixed = held[short] - per_token * short
        rows.append((tag, per_token, fixed))
        print("%-10s %7.3f KiB/token over %d attention layers, %7.2f MiB fixed "
              "recurrent state, max logit gap %.4f"
              % (tag, per_token / 1024, len(attention), fixed / 1024 / 1024, gap),
              flush=True)
        del model
        torch.cuda.empty_cache()

    if len(rows) == 2:
        print("\nper-token cache measured: %.2fx" % (rows[0][1] / max(rows[1][1], 1)))
        for context in (32768, 262144):
            print("  at %7d tokens: %6.2f GiB -> %6.2f GiB"
                  % (context, (rows[0][1] * context + rows[0][2]) / 1024 ** 3,
                     (rows[1][1] * context + rows[1][2]) / 1024 ** 3))
        print("\nthe logit gap is the cached path against one whole-sequence forward;")
        print("bf16 over this many positions will not be bitwise, but it should be small.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
