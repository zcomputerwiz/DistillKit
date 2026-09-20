"""Does a borrowing layer want a small channel of its own, and what does it cost.

A Reuse layer builds its keys and values from a donor's latent through `kv_b_proj`, and
the fit is visibly worse for it: r2 of 0.60 to 0.85 against 0.87 to 0.97 for a layer that
owns its latent. The proposal is to give each borrower a narrow per-token encoding of its
*own* hidden state and read keys and values from both:

    K_l = W^K_l c + U^K_l e_l,      V_l = W^V_l c + U^V_l e_l

with `e_l` much narrower than `c`. Setting its width to zero recovers pure borrowing, so
the comparison is exact rather than approximate.

What is already known and bounds the answer: trained, borrowing costs 0.0154 nats against
all-full and saves half the cache. So the gap this is meant to close is small, and the
channel is not free -- a borrower that caches `width` numbers a token stops caching
nothing, which is where borrowing's second halving comes from.

This measures the fit rather than a training run, because the fit is what the conversion
hands training and it takes seconds instead of hours. A width that cannot close the r2 gap
will not close the nats gap either; one that does earns a real arm.

    python scratch/dense_gr/correction_channel.py --source <checkpoint>
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

from convert_full import (STORE, calibration_windows, open_split,  # noqa: E402
                          solve, whiten)


def capture(model, full, store, vocab, count, length, device):
    """Each full-attention layer's input, and the keys and values it produced."""
    book = {index: [] for index in full}
    head_dim = model.config.head_dim
    rope = int(head_dim * model.config.rope_parameters["partial_rotary_factor"])
    groups = model.config.num_attention_heads // model.config.num_key_value_heads

    def hook(index):
        def inner(module, args, kwargs):
            states = kwargs.get("hidden_states", args[0] if args else None)
            shape = (*states.shape[:-1], -1, head_dim)
            key = module.k_norm(module.k_proj(states).view(shape))
            value = module.v_proj(states).view(shape)
            book[index].append((
                states.reshape(-1, states.shape[-1]).double(),
                torch.cat([
                    key[..., rope:].repeat_interleave(groups, -2).flatten(-2),
                    value.repeat_interleave(groups, -2).flatten(-2)],
                    dim=-1).reshape(states.shape[0] * states.shape[1], -1).double()))
        return inner

    handles = [model.model.layers[i].self_attn.register_forward_pre_hook(
        hook(i), with_kwargs=True) for i in full]
    with torch.no_grad():
        for ids in calibration_windows(store, vocab, count, length, device):
            model(input_ids=ids, use_cache=False)
    for handle in handles:
        handle.remove()
    return {i: tuple(torch.cat(part) for part in zip(*rows)) for i, rows in book.items()}


def explained(source, target):
    """Share of the target a least-squares read of `source` reproduces."""
    fitted = source @ solve(source, target).T
    return float(1 - (fitted - target).pow(2).sum() / target.pow(2).sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--latent", type=int, default=384)
    parser.add_argument("--widths", type=int, nargs="+", default=[0, 16, 32, 64, 128])
    parser.add_argument("--calibrate", type=int, default=16)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = Qwen35WidenedForCausalLM.from_pretrained(
        args.source, dtype=torch.bfloat16).to(device).eval()
    full = [i for i, kind in enumerate(model.config.layer_types)
            if "linear" not in str(kind)]
    vocab = model.config.vocab_size
    book = capture(model, full, args.store, vocab, args.calibrate, args.length, device)
    del model
    torch.cuda.empty_cache()
    print("captured %d tokens per layer, %d full-attention layers\n"
          % (next(iter(book.values()))[0].shape[0], len(full)))

    # The donor is the layer in front; every later one borrows from it, which is the
    # hardest case in the assignment sweep and the one a correction channel is for.
    donor = full[0]
    inputs, target = book[donor]
    whitener = whiten(inputs.T @ inputs)
    _, _, right = torch.linalg.svd((target.T @ inputs) @ whitener, full_matrices=False)
    encoder = right[:args.latent] @ whitener
    shared = inputs @ encoder.T

    header = "  ".join("%7s" % ("e=%d" % w) for w in args.widths)
    print("%-10s %s" % ("borrower", header))
    print("-" * (11 + 9 * len(args.widths)))
    for borrower in full[1:]:
        own, wanted = book[borrower]
        cells = []
        for width in args.widths:
            if width == 0:
                cells.append(explained(shared, wanted))
                continue
            # The borrower's own narrow encoding, built the same way its donor's was: the
            # best rank-`width` linear summary of its input for what it has to produce.
            local = whiten(own.T @ own)
            _, _, basis = torch.linalg.svd((wanted.T @ own) @ local, full_matrices=False)
            correction = own @ (basis[:width] @ local).T
            cells.append(explained(torch.cat([shared, correction], dim=-1), wanted))
        print("%-10d %s" % (borrower, "  ".join("%7.4f" % c for c in cells)))

    print("\ne=0 is pure borrowing. A width that does not lift the share is a cache cost")
    print("for nothing; borrowing's second halving comes from caching none at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
