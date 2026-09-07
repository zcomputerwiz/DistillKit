"""Gate 3, final form: real IQ4_NL rows through the real student, CPU-only.

Everything up to now exercised the sidecar with synthetic rows or with the
projection zeroed, which proves dormancy but not that the *table* path is live.
A zero-init projection makes a dropped feature tensor and a working one produce
identical logits, so dormancy alone cannot distinguish "wired up" from "silently
discarded". This drives the actual 28.8 GB table through SidecarDataCollator into
Qwen35SidecarForCausalLM and then perturbs the projection to force the features to
matter.

CPU-only and deliberately so: the dev box's GPUs hold the user's own model, and
loading here would evict it. torch.cuda.is_available is stubbed in-process
because CUDA_VISIBLE_DEVICES=-1 segfaults this Windows build.

Checks, in order of what they would catch:
  1. Rows gathered from the real table dequantize to finite, non-degenerate
     features with the magnitude the table's statistics predict.
  2. Different tokens produce different features (a broken hash or a constant
     gather would still be "finite and non-degenerate").
  3. With the projection at its zero init, sidecar_enabled=True and False give
     bit-identical logits -- dormancy holds with real features flowing.
  4. With the projection perturbed, logits change and stay finite -- the feature
     path is live end to end, not dropped somewhere between collator and residual.
  5. A real loss puts gradient on W_side_proj, so stage 1 can actually train it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_GGUF = os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "models--unsloth--Qwen3.8-Flash-Next-GGUF", "snapshots",
    "38bb39ee97821de2c9009abb7e93950eec396e66", "UD-IQ4_XS",
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf",
)
DEFAULT_STUDENT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "student-hf")
)

PROMPTS = [
    "The lighthouse keeper counted the ships passing the harbour at dawn.",
    "A frozen n-gram table supplies lexical features the backbone never learned.",
]


class _PadCollator:
    """Minimal base collator: pad to the longest sequence, right side."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        longest = max(len(f["input_ids"]) for f in features)
        ids, mask = [], []
        for f in features:
            n = len(f["input_ids"])
            ids.append(list(f["input_ids"]) + [self.pad_token_id] * (longest - n))
            mask.append([1] * n + [0] * (longest - n))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", default=DEFAULT_GGUF)
    parser.add_argument("--student", default=DEFAULT_STUDENT)
    parser.add_argument("--resident", action="store_true",
                        help="copy the 28.8 GB table into process RAM instead of mmap")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--report")
    args = parser.parse_args()

    # Must precede any transformers import that probes for accelerators.
    torch.cuda.is_available = lambda: False
    print("CUDA disabled in-process:", not torch.cuda.is_available())

    from transformers import AutoTokenizer

    from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM
    from distillkit.ngram_hash import NGramHasher
    from distillkit.ngram_table import GGUFNGramTable
    from distillkit.sidecar_collator import SidecarDataCollator

    results: dict[str, object] = {"gguf": args.gguf, "student": args.student, "dtype": args.dtype}

    table = GGUFNGramTable(args.gguf)
    print("==", table)
    if args.resident:
        print(f"   loaded resident in {table.load_resident():.1f}s")

    hasher = NGramHasher()
    tokenizer = AutoTokenizer.from_pretrained(args.student)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    collator = SidecarDataCollator(_PadCollator(pad_id), table, hasher)

    batch = collator([{"input_ids": tokenizer(p)["input_ids"]} for p in PROMPTS])
    ids, ngram_raw = batch["input_ids"], batch["ngram_raw"]
    print(f"   batch input_ids {tuple(ids.shape)}  ngram_raw {tuple(ngram_raw.shape)} {ngram_raw.dtype}")
    results["batch_shape"] = list(ids.shape)
    results["ngram_raw_shape"] = list(ngram_raw.shape)

    # --- 1/2: are the real features sane, and do they actually depend on tokens? ---
    from distillkit.ngram_table import IQ4NLDequant

    features = IQ4NLDequant(out_dtype=torch.float32)(ngram_raw).flatten(-2)
    l2 = features.norm(dim=-1)
    flat = features.reshape(-1, features.shape[-1])
    unique_frac = torch.unique(flat, dim=0).shape[0] / flat.shape[0]
    print(f"\n[1] features {tuple(features.shape)}  finite={torch.isfinite(features).all().item()}"
          f"  l2 mean {l2.mean():.4f} min {l2.min():.4f} max {l2.max():.4f}")
    print(f"[2] distinct feature vectors across positions: {unique_frac:.2%}")
    assert torch.isfinite(features).all(), "real table produced non-finite features"
    assert features.shape[-1] == 2560, features.shape
    assert (features != 0).any(dim=-1).all(), "some positions gathered an all-zero (padding) row"
    # 16 heads x 160 dims of per-element std ~0.0076 predicts l2 ~ sqrt(2560)*0.0076 ~ 0.38
    assert 0.2 < l2.mean() < 0.7, f"feature magnitude {l2.mean():.4f} outside the table's statistics"
    assert unique_frac > 0.9, "features barely vary across positions -- gather or hash is degenerate"
    results["feature_l2_mean"] = float(l2.mean())
    results["feature_unique_fraction"] = float(unique_frac)

    # --- load the real student ---------------------------------------------
    dtype = getattr(torch, args.dtype)
    t0 = time.perf_counter()
    model = Qwen35SidecarForCausalLM.from_pretrained(args.student, dtype=dtype).eval()
    print(f"\n   student loaded in {time.perf_counter()-t0:.1f}s "
          f"({sum(p.numel() for p in model.parameters())/1e9:.2f}B params, {args.dtype})")
    sidecar = model.model.layers[model.config.sidecar_layer_index].sidecar
    assert sidecar is not None
    assert sidecar.W_side_proj.weight.abs().max() == 0, "W_side_proj did not load at its zero init"

    kw = dict(input_ids=ids, attention_mask=batch["attention_mask"], return_dict=True)

    # --- 3: dormancy holds with real features flowing ----------------------
    t0 = time.perf_counter()
    with torch.no_grad():
        on = model(**kw, ngram_raw=ngram_raw, sidecar_enabled=True).logits
        off = model(**kw, sidecar_enabled=False).logits
    print(f"\n[3] dormant sidecar vs disabled: bit-identical={torch.equal(on, off)}"
          f"  (2 forwards in {time.perf_counter()-t0:.1f}s)")
    assert torch.isfinite(on).all(), "logits non-finite with the real table attached"
    assert torch.equal(on, off), "zero-init projection changed the logits -- dormancy broken"
    results["dormant_bit_identical"] = True

    # --- 4: the feature path is live ---------------------------------------
    with torch.no_grad():
        torch.manual_seed(0)
        sidecar.W_side_proj.weight.normal_(std=1e-3)
        perturbed = model(**kw, ngram_raw=ngram_raw, sidecar_enabled=True).logits
        # Rows for DIFFERENT tokens, same shape and same quantization statistics.
        # Zeroed rows would only re-prove dormancy: they dequantize to zero features,
        # so W @ 0 reproduces the dormant logits exactly. Real-but-wrong rows are what
        # distinguishes "these specific rows reached the residual" from "some nonzero
        # tensor did" -- a permuted or off-by-one gather passes the zero-row version.
        other_ids = ids.flip(1).roll(1, dims=1)
        other_raw = collator([{"input_ids": row.tolist()} for row in other_ids])["ngram_raw"]
        assert not torch.equal(other_raw, ngram_raw), "control rows are identical to the real ones"
        wrong = model(**kw, ngram_raw=other_raw, sidecar_enabled=True).logits
    delta = (perturbed.float() - on.float()).abs().max().item()
    wrong_delta = (wrong.float() - perturbed.float()).abs().max().item()
    print(f"[4] perturbed projection: max|dlogit| vs dormant {delta:.5f}"
          f"  | these rows vs other-token rows {wrong_delta:.5f}")
    assert torch.isfinite(perturbed).all(), "perturbed sidecar produced non-finite logits"
    assert delta > 1e-4, "perturbing W_side_proj did not move the logits -- features are dropped"
    assert wrong_delta > 1e-4, (
        "rows for different tokens give the same logits -- the gather is token-independent"
    )
    results["perturbed_logit_delta"] = delta
    results["other_token_rows_delta"] = wrong_delta

    # --- 5: stage 1 can train it -------------------------------------------
    with torch.no_grad():
        sidecar.W_side_proj.weight.zero_()
    model.freeze_backbone()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    out = model(**kw, ngram_raw=ngram_raw, labels=ids, sidecar_enabled=True)
    out.loss.backward()
    grad = sidecar.W_side_proj.weight.grad
    print(f"[5] stage-1 trainable tensors: {len(trainable)}  loss {out.loss.item():.4f}"
          f"  W_side_proj grad norm {grad.float().norm().item():.6f}")
    assert grad is not None and grad.float().norm().item() > 0, (
        "no gradient on W_side_proj -- zero init would stay zero forever"
    )
    assert all("sidecar" in n or n.startswith("distillation_projections.") for n in trainable), (
        f"stage-1 freeze left non-sidecar parameters trainable: {trainable[:5]}"
    )
    results["stage1_trainable"] = trainable
    results["loss"] = out.loss.detach().item()
    results["w_side_proj_grad_norm"] = float(grad.float().norm())

    print("\nGATE 3 (real table, final student): PASS")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print("wrote", args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
