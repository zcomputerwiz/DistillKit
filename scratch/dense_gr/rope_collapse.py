"""What the conversion's two approximations cost, measured on the source alone.

The MLA conversion costs the same at latent rank 128 as at 384 -- 1.9106 against 1.9097
on the toy -- and rank invariance says the damage is not in the latent. There is exactly
one other approximation in the conversion, and it is not ranked: MLA carries a single
decoupled rotary key, so `source_capture` collapses the source's per-key-head rotary
slices to their mean. A model with two key/value heads has two different rotary slices,
and one mean has to stand for both.

That claim can be tested without converting anything. Reach into the source and apply each
approximation to it in isolation:

    baseline    the model as it is
    rope mean   the rotary slice of every key head replaced by the mean across heads
    content SVD the content slice replaced by its own rank-r reconstruction

The first tells us what the collapse costs. The second tells us what the latent costs. If
the collapse alone reproduces the conversion's gap, then the conversion is not fitting
badly -- it is discarding something no fit can recover, and the latent rank was never the
variable worth sweeping.
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import triton_shim  # noqa: F401,E402

from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402

from convert_full import STORE, heldout  # noqa: E402


@contextlib.contextmanager
def intervene(model, rope_dim, kind, rank):
    """Apply one of the conversion's approximations inside the source's key path.

    `k_norm` sits between the key projection and the rotary, which is exactly where the
    conversion takes its targets, so wrapping it applies the approximation to the same
    tensor the fit would have seen.
    """
    full = [i for i, k in enumerate(model.config.layer_types) if "linear" not in str(k)]
    saved = {}
    for index in full:
        norm = model.model.layers[index].self_attn.k_norm
        original = norm.forward
        saved[index] = (norm, original)

        def wrapped(hidden_states, _original=original):
            key = _original(hidden_states)
            if kind == "rope":
                # One shared rotary slice standing in for every key head's own.
                shared = key[..., :rope_dim].mean(-2, keepdim=True)
                return torch.cat(
                    [shared.expand_as(key[..., :rope_dim]), key[..., rope_dim:]], dim=-1)
            content = key[..., rope_dim:]
            shape = content.shape
            flat = content.reshape(-1, shape[-2] * shape[-1]).float()
            # The latent is a rank-r bottleneck on the content, which is what an SVD
            # truncation of the same width costs at best.
            u, s, v = torch.linalg.svd(flat - flat.mean(0), full_matrices=False)
            keep = min(rank, s.numel())
            low = (u[:, :keep] * s[:keep]) @ v[:keep] + flat.mean(0)
            return torch.cat([key[..., :rope_dim], low.reshape(shape).to(key.dtype)],
                             dim=-1)
        norm.forward = wrapped
    try:
        yield
    finally:
        for norm, original in saved.values():
            norm.forward = original


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=384)
    parser.add_argument("--evaluate", type=int, default=64)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = Qwen35WidenedForCausalLM.from_pretrained(
        args.source, dtype=torch.bfloat16).to(device).eval()
    head_dim = model.config.head_dim
    rope_dim = int(head_dim * model.config.rope_parameters["partial_rotary_factor"])
    vocab = model.config.vocab_size
    print("head %d, rotary slice %d, %d key/value heads"
          % (head_dim, rope_dim, model.config.num_key_value_heads))

    base = heldout(model, args.store, vocab, args.evaluate, args.length, device)
    print("\nbaseline          %.4f" % base, flush=True)
    with intervene(model, rope_dim, "rope", args.rank):
        collapsed = heldout(model, args.store, vocab, args.evaluate, args.length, device)
    print("rotary collapsed  %.4f  %+.4f" % (collapsed, collapsed - base), flush=True)
    with intervene(model, rope_dim, "content", args.rank):
        truncated = heldout(model, args.store, vocab, args.evaluate, args.length, device)
    print("content rank %-4d %.4f  %+.4f" % (args.rank, truncated, truncated - base))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
