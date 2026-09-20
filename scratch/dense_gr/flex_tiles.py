"""Which FlexAttention tiles fit this card, at the head width CSA2 actually hands it.

The 2B conversion died in the compiler, not in the model:

    No valid triton configs. OutOfMemoryError: out of resource:
    triton_tem_fused_flex_attention_0  Required: 167936  Hardware limit: 101376

Inductor rounds the key/query head up to a power of two before it allocates anything,
then picks a tile from a table keyed on the unrounded width without checking the result
against the device. There is no fallback to a smaller tile -- the config that does not fit
is dropped and the choice list empties. So the tile has to be chosen from outside, and
this is what it is chosen from.

Run it whenever `head_dim`, `csa2_index_dim` or the card changes, and copy the winners
into `Qwen35SparseLatentAttention._kernel_options`:

    python scratch/dense_gr/flex_tiles.py --device cuda:1

The two widths that matter are the head on its own, which is the dense path, and the head
plus the router's index columns, which is the sparse one.
"""
from __future__ import annotations

import argparse
import itertools
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import triton_shim  # noqa: F401,E402

from torch.nn.attention.flex_attention import (create_block_mask,  # noqa: E402
                                               flex_attention)

REQUIRED = re.compile(r"Required:\s*(\d+)")
LIMIT = re.compile(r"Hardware limit:\s*(\d+)")


def rounded(width):
    """What Inductor pads the head to: `next_power_of_two` in its flex common."""
    return 1 << (width - 1).bit_length()


def build(batch, heads, length, width, value_width, device):
    shape = (batch, heads, length, width)
    return (torch.randn(shape, device=device, dtype=torch.bfloat16),
            torch.randn(shape, device=device, dtype=torch.bfloat16),
            torch.randn((batch, heads, length, value_width), device=device,
                        dtype=torch.bfloat16))


def attempt(shape, width, options, mask, device, repeats, backward=False):
    """Compile and launch once. ('ok', ms), ('oom', asked, limit) or ('err', line).

    The backward has its own table and its own failure. Inductor keys it on head_dim and
    hands SM86 a 64x64x64x64 tile at 256 and a 16x16x16x16 one above it, so a layer that
    carries the router's columns fits and one that does not -- a reuse layer, which has no
    columns to carry -- asks 168960 of 101376 and never compiles.
    """
    torch._dynamo.reset()
    query, key, value = build(*shape[:3], width, shape[3], device)
    if backward:
        for tensor in (query, key, value):
            tensor.requires_grad_(True)
    compiled = torch.compile(flex_attention, dynamic=False)

    def once():
        out = compiled(query, key, value, block_mask=mask, kernel_options=options)
        if backward:
            out.sum().backward()

    try:
        once()
        torch.cuda.synchronize()
        began = time.perf_counter()
        for _ in range(repeats):
            once()
        torch.cuda.synchronize()
        return ("ok", (time.perf_counter() - began) * 1000.0 / repeats)
    except Exception as error:  # noqa: BLE001 - the message is the measurement
        text = "%s" % error
        asked, limit = REQUIRED.search(text), LIMIT.search(text)
        if asked and limit:
            return ("oom", int(asked.group(1)), int(limit.group(1)))
        return ("err", text.strip().splitlines()[0][:96])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--index-dim", type=int, default=64,
                        help="csa2_index_dim, the router's extra query and key columns")
    parser.add_argument("--block-size", type=int, default=128,
                        help="csa2_block_size; BLOCK_M and BLOCK_N have to divide it")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--backward", action="store_true",
                        help="sweep the backward tile instead of the forward one")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    major, minor = torch.cuda.get_device_capability(device)
    limit = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
    print("%s  sm_%d%d  %d bytes of shared memory per block"
          % (torch.cuda.get_device_name(device), major, minor, limit))

    widths = [args.head_dim, args.head_dim + args.index_dim]
    print("\nhead widths and what Inductor pads them to")
    for width in widths:
        print("  %3d -> %3d%s" % (width, rounded(width),
                                  "" if rounded(width) == width else "   (not a power of two)"))

    window = args.window

    def local_causal(b, h, q, kv):
        return (kv <= q) & (q - kv < window)

    mask = create_block_mask(local_causal, args.heads and 1, args.heads, args.length,
                             args.length, device=str(device), BLOCK_SIZE=args.block_size)
    shape = (1, args.heads, args.length, args.head_dim)

    print("\n%-6s %-22s %-8s %s" % ("width", "tile", "outcome", "detail"))
    print("-" * 70)
    if args.backward:
        # The forward tile has to come along, or the graph dies there and every row
        # reports the forward's number instead of the backward's. These are the measured
        # forward winners, keyed on the rounded head in bytes the way _kernel_options is.
        def forward_for(width):
            span = (1 << (width - 1).bit_length()) * 2
            stages = 3 if span <= 512 else 1
            return {"fwd_BLOCK_M": 16, "fwd_BLOCK_N": 32, "fwd_num_stages": stages}

        # One square tile for both halves of the backward, which is what every entry in
        # Inductor's own table does, and the stage count on top.
        grid = [None] + [{"bwd_BLOCK_M1": b, "bwd_BLOCK_N1": b,
                          "bwd_BLOCK_M2": b, "bwd_BLOCK_N2": b, "bwd_num_stages": s}
                         for b, s in itertools.product((64, 32, 16), (3, 2, 1))]
    else:
        grid = [None] + [{"BLOCK_M": m, "BLOCK_N": n, "num_stages": s}
                         for m, n, s in itertools.product((128, 64, 32, 16), (64, 32, 16),
                                                          (3, 2, 1))]
    fitting = {}
    for width in widths:
        for options in grid:
            if options is None:
                label = "default"
            elif args.backward:
                label = "block%-3d stages%d" % (options["bwd_BLOCK_M1"],
                                                options["bwd_num_stages"])
            else:
                label = "M%-3d N%-3d stages%d" % (options["BLOCK_M"], options["BLOCK_N"],
                                                  options["num_stages"])
            passed = options
            if args.backward and options is not None:
                passed = dict(options)
                passed.update(forward_for(width))
            outcome = attempt(shape, width, passed, mask, device, args.repeats,
                              backward=args.backward)
            if outcome[0] == "ok":
                detail = "%.3f ms" % outcome[1]
                if options is not None:
                    fitting.setdefault(width, []).append((outcome[1], label, options))
            elif outcome[0] == "oom":
                detail = "asks %d of %d" % (outcome[1], outcome[2])
            else:
                detail = outcome[1]
            print("%-6d %-22s %-8s %s" % (width, label, outcome[0], detail), flush=True)

    print("\nfastest tile that fits")
    for width in widths:
        if width in fitting:
            best = min(fitting[width])
            print("  width %3d  %s  %.3f ms  ->  %s" % (width, best[1], best[0], best[2]))
        else:
            print("  width %3d  nothing in the grid fits; try BLOCK_N 8" % width)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
