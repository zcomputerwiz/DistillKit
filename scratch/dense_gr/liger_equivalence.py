"""Does the Liger swap compute the same model?

The swap replaces `Qwen3_5MLP.forward` with a fused SiLU-multiply and every
`Qwen3_5RMSNorm` with `LigerRMSNorm(offset=1.0, casting_mode="gemma")`. Both are claimed
to match Qwen3.5's conventions -- the norm stores a deviation and applies `1 + weight` in
fp32 before casting back -- but "claimed to match" is how the `2 * weight` trap in the GR
conversion nearly shipped. So compare the modules in isolation, then the whole model's
logits, against the stock implementation with identical weights.

    CUDA_VISIBLE_DEVICES=0 python scratch/dense_gr/liger_equivalence.py
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

from benchmark import apply_liger, build  # noqa: E402

from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402


def difference(left, right):
    gap = (left.float() - right.float()).abs()
    scale = right.float().abs().max().clamp(min=1e-12)
    return {"max_abs": float(gap.max()), "max_rel": float(gap.max() / scale),
            "bitwise": bool(torch.equal(left, right))}


def norm_only(hidden, dtype, device):
    """The norm in isolation, where a convention mismatch would be unmissable."""
    from liger_kernel.transformers import LigerRMSNorm

    torch.manual_seed(3)
    stock = Qwen3_5RMSNorm(hidden, eps=1e-6).to(device=device, dtype=dtype)
    with torch.no_grad():
        stock.weight.uniform_(-0.4, 0.5)
    fused = LigerRMSNorm(hidden, eps=1e-6, offset=1.0, casting_mode="gemma",
                         init_fn="zeros").to(device=device, dtype=dtype)
    fused.weight.data.copy_(stock.weight.data)
    x = torch.randn(4, 128, hidden, device=device, dtype=dtype) * 3
    return difference(fused(x), stock(x))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--vocab", type=int, default=32_768)
    parser.add_argument("--output", type=Path,
                        default=Path("scratch/dense_gr/liger-equivalence.json"))
    args = parser.parse_args()

    device, dtype = "cuda", torch.bfloat16
    report = {"hidden": args.hidden, "layers": args.layers, "vocab": args.vocab,
              "dtype": str(dtype)}

    report["rmsnorm_isolated"] = norm_only(args.hidden, dtype, device)
    report["rmsnorm_isolated_fp32"] = norm_only(args.hidden, torch.float32, device)

    config = build(args.hidden, args.layers, args.vocab,
                   attn_implementation="flash_attention_2")
    torch.manual_seed(0)
    stock = Qwen35WidenedForCausalLM(config).to(device=device, dtype=dtype).eval()
    fused = copy.deepcopy(stock)
    report["swapped"] = apply_liger(fused, config)

    tokens = torch.randint(0, args.vocab, (2, 256), device=device)
    mask = torch.ones_like(tokens)
    with torch.inference_mode():
        a = stock(input_ids=tokens, attention_mask=mask, use_cache=False).logits
        b = fused(input_ids=tokens, attention_mask=mask, use_cache=False).logits
    report["logits"] = difference(b, a)
    report["argmax_agreement"] = float((a.argmax(-1) == b.argmax(-1)).float().mean())

    print(json.dumps(report, indent=2))
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
