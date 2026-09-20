"""Compare the query, key and value two models hand to attention, at the same layer.

The conversion's own approximations cost 0.036 nats when applied to the source directly,
and the conversion costs 6.38. So the gap is not the design -- something in the assembled
tensors is wrong. `key_r2` cannot see it, because it scores the content slice the fit
targets and never the rotary slice that carries position.

This registers an attention interface that records what it was called with, puts both
models on it, and compares the three tensors piece by piece: the rotary half of the key
against the source's, the content half against the source's, and the query and value
whole. Whichever piece is wrong is the bug, and the halves are reported apart because
they are fitted by different code and fail for different reasons.
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

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402
from transformers.integrations.sdpa_attention import (  # noqa: E402
    sdpa_attention_forward)

from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402

from convert_full import STORE, open_split  # noqa: E402

CAUGHT = {}


def probe(module, query, key, value, attention_mask, **kwargs):
    CAUGHT.setdefault(id(module.__class__), {})[module.layer_idx] = (
        query.detach().float().cpu(), key.detach().float().cpu(),
        value.detach().float().cpu())
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


def agreement(reference, other):
    a, b = reference.flatten(), other.flatten()
    residual = float(((a - b) ** 2).sum())
    total = float(((a - a.mean()) ** 2).sum())
    return 1.0 - residual / max(total, 1e-12), float(b.norm()) / max(float(a.norm()), 1e-12)


def run(path, ids, device, tag):
    model = Qwen35WidenedForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16).to(device).eval()
    model.config._attn_implementation = "probe"
    for layer in model.model.layers:
        attention = getattr(layer, "self_attn", None)
        if attention is not None:
            attention.config._attn_implementation = "probe"
    CAUGHT.clear()
    with torch.no_grad():
        model.model(input_ids=ids.to(device),
                    attention_mask=torch.ones_like(ids).to(device), use_cache=False)
    caught = {}
    for per_class in CAUGHT.values():
        caught.update(per_class)
    print("%s: caught %d layers" % (tag, len(caught)))
    config = model.config
    del model
    torch.cuda.empty_cache()
    return caught, config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    ALL_ATTENTION_FUNCTIONS["probe"] = probe

    stream = open_split(args.store, "heldout", 248320)
    ids = torch.from_numpy(
        np.array(stream[:args.length], dtype=np.int64).reshape(1, args.length))

    theirs, source_config = run(args.source, ids, args.device, "source")
    mine, _ = run(args.converted, ids, args.device, "converted")

    head_dim = source_config.head_dim
    rope_dim = int(head_dim * source_config.rope_parameters["partial_rotary_factor"])
    print("\nhead %d, rotary slice %d\n" % (head_dim, rope_dim))
    print("%-7s %18s %18s %18s %18s"
          % ("layer", "query r2/scale", "key rope r2/scale", "key content r2/sc",
             "value r2/scale"))
    print("-" * 82)
    for index in sorted(set(theirs) & set(mine)):
        (tq, tk, tv), (mq, mk, mv) = theirs[index], mine[index]
        # The source hands the interface its two key/value heads and lets GQA expand
        # them; MLA hands over one per query head already. Expand the source the way
        # `repeat_kv` does so the two are the same shape and the same head order.
        if tk.shape[1] != mk.shape[1]:
            groups = mk.shape[1] // tk.shape[1]
            tk = tk.repeat_interleave(groups, dim=1)
            tv = tv.repeat_interleave(groups, dim=1)
        cells = [agreement(tq, mq), agreement(tk[..., :rope_dim], mk[..., :rope_dim]),
                 agreement(tk[..., rope_dim:], mk[..., rope_dim:]), agreement(tv, mv)]
        print("%-7d %s" % (index,
                           " ".join("%10.4f %7.4f" % cell for cell in cells)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
