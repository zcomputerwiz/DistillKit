"""Gate 3 at full size: stock Qwen3.5-4B text decoder vs Qwen35SidecarForCausalLM.

Loads the real student-stock VLM checkpoint (model.language_model.* layout, visual +
mtp keys excluded) into both classes on CPU in bf16 and requires bit-identical
logits and hidden states with the zero-init sidecar active. Reports any divergence
with sign-of-zero awareness: adding a mathematically-zero vector can flip -0.0 to
+0.0 under IEEE rules, which is numerically identical but not bit-identical; real
divergence (nonzero difference) is a bug in the subclass.

No GPU: torch.cuda.is_available is pinned False before any model work.
"""

import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch

torch.cuda.is_available = lambda: False
assert not torch.cuda.is_available()

STOCK_PATH = r"D:\DeepThought\Projects\HybridModel\student-stock"


def compare(tag, a, b):
    if torch.equal(a, b):
        print(f"{tag}: BIT-IDENTICAL  {tuple(a.shape)} {a.dtype}")
        return True
    af, bf = a.float(), b.float()
    an, bn = a.isnan(), b.isnan()
    nan_mismatch = int((an != bn).sum())
    finite = ~(an | bn)
    diff = (af - bf).abs()
    nz = int((finite & (diff > 0)).sum())
    # sign-of-zero flips: numerically equal but different bit patterns
    bit_flips = int(((a.view(torch.int16) != b.view(torch.int16)) & finite & (af == bf)).sum())
    max_diff = diff[finite].max().item() if nz else 0.0
    print(
        f"{tag}: DIVERGES  {tuple(a.shape)} | nonzero diffs: {nz} | nan mismatches: {nan_mismatch} | "
        f"max abs diff: {max_diff:.6g} | sign-of-zero bit flips: {bit_flips}"
    )
    if nz:
        idx = torch.argmax(torch.where(finite, diff, torch.zeros_like(diff)))
        flat_a, flat_b = a.flatten(), b.flatten()
        print(f"  first real divergence at flat index {idx.item()}: {flat_a[idx].item()} vs {flat_b[idx].item()}")
    return nan_mismatch == 0 and nz == 0


def main():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from distillkit.models import Qwen35SidecarForCausalLM

    torch.manual_seed(20260906)
    ids = torch.randint(0, 248077, (1, 256))

    print("Loading stock text decoder ...")
    stock = Qwen3_5ForCausalLM.from_pretrained(STOCK_PATH, torch_dtype=torch.bfloat16)
    stock.eval()

    print("Loading sidecar student ...")
    custom, info = Qwen35SidecarForCausalLM.from_pretrained(
        STOCK_PATH, torch_dtype=torch.bfloat16, output_loading_info=True
    )
    custom.eval()

    print(f"missing keys: {len(info['missing_keys'])}  unexpected: {len(info['unexpected_keys'])}")
    non_sidecar_missing = [k for k in info["missing_keys"] if ".sidecar." not in k]
    print(f"non-sidecar missing: {non_sidecar_missing[:10]}")
    assert not info["unexpected_keys"], "unexpected keys on VLM-prefixed load"
    assert not non_sidecar_missing, "stock weights did not all load"

    # sidecar must be dormant at init
    sidecar = custom.model.layers[1].sidecar
    assert torch.count_nonzero(sidecar.W_side_proj.weight) == 0
    assert all(torch.count_nonzero(b.weight) == 0 for b in sidecar.gated_residual.branches)

    # raw rows: 16 heads x 90 bytes (IQ4_NL, 5 blocks of 18) per token.
    # Bytes 0-1 of each block are an fp16 scale: random byte pairs can decode to
    # NaN/inf scales, so pin them to a finite value like the table's real rows.
    g = torch.Generator().manual_seed(7)
    raw = torch.randint(0, 256, (1, 256, 16, 90), dtype=torch.uint8, generator=g)
    scale = torch.full((1, 256, 16, 1), 0.001, dtype=torch.float16)
    raw.view(1, 256, 16, 5, 18)[..., :2] = scale.view(torch.uint8).view(1, 256, 16, 1, 2).expand(1, 256, 16, 5, 2)

    with torch.no_grad():
        out_s = stock(ids, output_hidden_states=True)
        out_c = custom(ids, ngram_raw=raw, output_hidden_states=True)

    ok = True
    ok &= compare("logits", out_s.logits, out_c.logits)
    assert len(out_s.hidden_states) == len(out_c.hidden_states) == 33
    for i, (hs, hc) in enumerate(zip(out_s.hidden_states, out_c.hidden_states)):
        if not torch.equal(hs, hc):
            ok &= compare(f"hidden_states[{i}]", hs, hc)
    print("\nGATE 3 FULL-SIZE PARITY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
