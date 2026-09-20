"""How much of a converted model's loss is the router, and how much is everything else.

A full-stack conversion of the 2B lands 6.1 nats above its source, which is far past what
the parts predict: the gated residual converts bitwise, and MLA at rank 384 measured
+0.037 on its own. The suspect is the router, because `csa2_top_k` counts tokens rather
than blocks -- 256 over a block size of 128 keeps two blocks of the eight a 1024-token
window has, and at conversion time it picks those two at random.

That is a claim about which term dominates, and it is cheap to settle. Read the same
checkpoint three ways:

    routed    as saved, the random router selecting two blocks in eight
    dense     `dense_routing`, every block open, so only MLA and the blend remain
    source    the model it was converted from

If dense lands near source then the conversion is sound and the gap is entirely the
untrained router, which is what `distill_indexer.py` is for. If dense is also far out,
something else is broken and distilling the router will not reach it.
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
from distillkit.models.qwen35.csa2 import (Qwen35SparseLatentAttention,  # noqa: E402
                                           dense_routing)

from convert_full import STORE, heldout  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--evaluate", type=int, default=64)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = Qwen35WidenedForCausalLM.from_pretrained(
        args.converted, dtype=torch.bfloat16).to(device).eval()
    vocab = model.config.vocab_size
    block = model.config.csa2_block_size
    blocks = args.length // block
    keep = max(1, min(model.config.csa2_top_k // block, blocks))
    print("%s\n  block %d, top_k %d tokens -> %d of %d blocks kept, local window %d"
          % (args.converted, block, model.config.csa2_top_k, keep, blocks,
             model.config.csa2_local_window))

    routed = heldout(model, args.store, vocab, args.evaluate, args.length, device)
    print("\n  routed   %.4f  blocks, the granularity training saw" % routed, flush=True)

    # The same model read the way it is served. Training selects whole blocks because a
    # BlockMask cannot express anything finer; a decode step selects positions. Same
    # budget, finer instrument, different function -- so whether the finer one is actually
    # better is a measurement rather than an assumption.
    layers = [l.self_attn for l in model.model.layers
              if isinstance(getattr(l, "self_attn", None), Qwen35SparseLatentAttention)]
    for layer in layers:
        layer._blocked = lambda seq, past, _l=layer: False
    gathered = heldout(model, args.store, vocab, args.evaluate, args.length, device)
    print("  gathered %.4f  positions, the granularity serving uses  %+.4f"
          % (gathered, gathered - routed), flush=True)
    for layer in layers:
        del layer._blocked

    with dense_routing(model):
        dense = heldout(model, args.store, vocab, args.evaluate, args.length, device)
    print("  dense    %.4f" % dense, flush=True)
    del model
    torch.cuda.empty_cache()

    if args.source is not None:
        stock = Qwen35WidenedForCausalLM.from_pretrained(
            args.source, dtype=torch.bfloat16).to(device).eval()
        origin = heldout(stock, args.store, vocab, args.evaluate, args.length, device)
        print("  source  %.4f" % origin)
        print("\nthe structure costs %+.4f, the random router costs %+.4f"
              % (dense - origin, routed - dense))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
