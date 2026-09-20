"""Where a converted model stops agreeing with the model it was converted from.

The MLA conversion costs about the same whatever the latent rank is -- 1.9106 nats at
128, 1.9097 at 384 on the toy, and 6.38 on the 2B. Rank invariance is the whole clue. If
the cost were the low-rank approximation, a third of the rank would cost visibly more; it
does not, so the cost is something constant that the rank cannot reach. Key and value fit
at r2 up to 0.97 and the model is still destroyed, which places the fault downstream of
the tensors the fit measures.

So walk both models over the same tokens and find the first place they part company:

    hidden states   at every layer boundary, which says *where*
    attention out   at each full-attention layer, which says whether it is the attention
    query/key/value and the gate, which says *what* inside it

A conversion that is merely approximate diverges a little at the first converted layer
and accumulates. One with a systematic error -- a rotary applied twice, a gate reading the
wrong half of a projection, a head layout that does not match -- diverges hard and at once.

    python scratch/dense_gr/mla_divergence.py \\
        --source D:/DeepThought/Projects/HybridModel/student-2b-hf \\
        --converted scratch/dense_gr/checkpoints-conv/student-2b-mla
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


def agreement(left, right):
    """Fraction of the reference's variance the other one keeps, and relative size."""
    a = left.detach().float().flatten().cpu()
    b = right.detach().float().flatten().cpu()
    residual = float(((a - b) ** 2).sum())
    total = float(((a - a.mean()) ** 2).sum())
    return (1.0 - residual / max(total, 1e-12),
            float(b.norm()) / max(float(a.norm()), 1e-12))


def capture(model, ids, device):
    """Hidden states per layer, plus each full-attention module's output."""
    caught = {}

    def hook(index):
        def inner(module, args, kwargs, output):
            caught[index] = (output[0] if isinstance(output, tuple) else output).to("cpu")
        return inner

    full = [i for i, kind in enumerate(model.config.layer_types)
            if "linear" not in str(kind)]
    handles = [model.model.layers[i].self_attn.register_forward_hook(
        hook(i), with_kwargs=True) for i in full]
    with torch.no_grad():
        out = model.model(input_ids=ids.to(device),
                          attention_mask=torch.ones_like(ids).to(device),
                          use_cache=False, output_hidden_states=True)
    for handle in handles:
        handle.remove()
    return [h.to("cpu") for h in out.hidden_states], caught, full


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-device", default="cuda:1")
    args = parser.parse_args()

    converted = Qwen35WidenedForCausalLM.from_pretrained(
        args.converted, dtype=torch.bfloat16).to(args.device).eval()
    source = Qwen35WidenedForCausalLM.from_pretrained(
        args.source, dtype=torch.bfloat16).to(args.source_device).eval()

    stream = open_split(args.store, "heldout", source.config.vocab_size)
    ids = torch.from_numpy(
        np.array(stream[:args.length], dtype=np.int64).reshape(1, args.length))

    reference, source_attn, full = capture(source, ids, args.source_device)
    theirs, converted_attn, _ = capture(converted, ids, args.device)

    print("full attention at %s\n" % full)
    print("%-7s %-9s %18s %18s" % ("layer", "kind", "hidden r2 / scale",
                                   "attention r2 / scale"))
    print("-" * 56)
    for index in range(len(reference) - 1):
        kind = "full" if index in full else "linear"
        r2, scale = agreement(reference[index + 1], theirs[index + 1])
        if index in source_attn and index in converted_attn:
            ar2, ascale = agreement(source_attn[index], converted_attn[index])
            attention = "%9.4f %8.4f" % (ar2, ascale)
        else:
            attention = "%18s" % ""
        print("%-7d %-9s %9.4f %8.4f %s" % (index, kind, r2, scale, attention))

    first = next((i for i in range(len(reference) - 1)
                  if agreement(reference[i + 1], theirs[i + 1])[0] < 0.99), None)
    print("\nfirst layer under r2 0.99: %s" % first)
    if first is not None and first in source_attn:
        mine, ref = converted_attn[first], source_attn[first]
        print("attention output at that layer: r2 %.4f, norm ratio %.4f"
              % agreement(ref, mine))
        print("  source    mean %+.5f  std %.5f" % (ref.float().mean(), ref.float().std()))
        print("  converted mean %+.5f  std %.5f"
              % (mine.float().mean(), mine.float().std()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
