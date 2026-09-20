"""Is preserving both key heads' rotary slices worth the cache it costs.

MLA carries one decoupled rotary key. This source has two key/value heads with two
different rotary slices, and the conversion replaces them with their mean. Measured on the
source directly that collapse costs +0.0361 nats, and I called it a floor no fit could
recover.

It is not a floor. After 30M tokens the whole structure costs +0.0249, below the figure
the collapse alone was supposed to impose, because the model adapts around a collapsed
rotary key rather than having to reconstruct it. So the question is not what the collapse
costs a frozen conversion -- it is whether keeping both slices is worth doubling that part
of the cache once training is allowed to respond.

Two things are measured here, and the second is the one that decides it:

    reconstruction   how much of each head's own rotary slice the mean keeps, and what a
                     fitted rank-`rope` map keeps instead. A mean that already captures
                     nearly all of it makes the rest moot.
    what it costs    both slices is `2 * rope` a token where the mean is `rope`, against
                     a 448-number cache. Long context is what the extra buys nothing for.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import triton_shim  # noqa: F401,E402

from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402

from convert_full import STORE, calibration_windows, solve  # noqa: E402


def capture(model, full, store, vocab, count, length, device):
    """Per key/value head rotary slices, before they are collapsed."""
    book = {index: [] for index in full}
    head_dim = model.config.head_dim
    rope = int(head_dim * model.config.rope_parameters["partial_rotary_factor"])

    def hook(index):
        def inner(module, args, kwargs):
            states = kwargs.get("hidden_states", args[0] if args else None)
            shape = (*states.shape[:-1], -1, head_dim)
            key = module.k_norm(module.k_proj(states).view(shape))
            book[index].append((
                states.reshape(-1, states.shape[-1]).double(),
                key[..., :rope].reshape(-1, key.shape[-2], rope).double()))
        return inner

    handles = [model.model.layers[i].self_attn.register_forward_pre_hook(
        hook(i), with_kwargs=True) for i in full]
    with torch.no_grad():
        for ids in calibration_windows(store, vocab, count, length, device):
            model(input_ids=ids, use_cache=False)
    for handle in handles:
        handle.remove()
    return {i: tuple(torch.cat(part) for part in zip(*rows)) for i, rows in book.items()}


def share(fitted, target):
    return float(1 - (fitted - target).pow(2).sum() / target.pow(2).sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--calibrate", type=int, default=16)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = Qwen35WidenedForCausalLM.from_pretrained(
        args.source, dtype=torch.bfloat16).to(device).eval()
    config = model.config
    head_dim = config.head_dim
    rope = int(head_dim * config.rope_parameters["partial_rotary_factor"])
    latent = 384
    full = [i for i, kind in enumerate(config.layer_types) if "linear" not in str(kind)]
    book = capture(model, full, args.store, config.vocab_size, args.calibrate,
                   args.length, device)
    del model
    torch.cuda.empty_cache()

    print("%d key/value heads, rotary slice %d wide\n" % (config.num_key_value_heads, rope))
    print("%-8s %10s %10s %10s" % ("layer", "mean", "fitted", "per-head"))
    print("-" * 42)
    for index in full:
        inputs, slices = book[index]
        heads = slices.shape[1]
        pooled = slices.mean(1, keepdim=True).expand_as(slices)
        # What one shared slice fitted from the hidden state keeps, against what the mean
        # of the heads keeps -- the conversion uses the mean, and a fit is the best a
        # single shared vector could do.
        flat = slices.reshape(slices.shape[0], heads * rope)
        shared = inputs @ solve(inputs, slices.mean(1)).T
        fitted = shared.unsqueeze(1).expand_as(slices)
        own = inputs @ solve(inputs, flat).T
        print("%-8d %10.4f %10.4f %10.4f"
              % (index, share(pooled, slices), share(fitted, slices),
                 share(own.reshape_as(slices), slices)))

    both, one = latent + 2 * rope, latent + rope
    print("\ncache per token per caching layer: %d with one slice, %d with both (+%.0f%%)"
          % (one, both, 100.0 * (both - one) / one))
    print("at 262,144 tokens over 6 layers that is %.2f GiB against %.2f"
          % (both * 6 * 2 * 262144 / 1024 ** 3, one * 6 * 2 * 262144 / 1024 ** 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
