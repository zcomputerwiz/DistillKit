"""Verify the GGUF->HF conversion of Qwen3.8-4B-BF16.gguf (student-hf/).

Three independent checks, all CPU-only (CUDA hidden in-process):

1. Gate 3 on the actual training base: stock class vs sidecar subclass must be
   bit-identical (logits + all hidden states) with the zero-init sidecar active.
2. Orientation oracle: per-row norm profiles of converted tensors vs the
   student-stock checkpoint (same family, pre-fine-tune). A transposed or
   same-shape-swapped tensor breaks the row-norm correlation even though check 1
   still passes (both classes load the same wrong weights and agree with each
   other).
3. Differential LM loss: converted fine-tune vs stock text decoder on the same
   real tokens. A broken conversion (e.g. ssm_alpha<->ssm_beta swap) sends the
   converted model's loss to ~ln(vocab) while stock stays sane.

No GPU: torch.cuda.is_available is pinned False before any model work.
"""

import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import numpy as np
import torch

torch.cuda.is_available = lambda: False
assert not torch.cuda.is_available()

CONVERTED_PATH = r"D:\DeepThought\Projects\HybridModel\student-hf"
STOCK_PATH = r"D:\DeepThought\Projects\HybridModel\student-stock"

PASSAGE = (
    "The lighthouse keeper climbed the spiral stairs each morning before dawn, "
    "checking the oil reservoirs and trimming the wick until the flame burned steady. "
    "Fog had rolled in from the north overnight, and the beams swept the water in slow "
    "arcs, catching the whitecaps far out beyond the breakwater. Down on the shore road, "
    "a fishing boat waited with its engines idling, its captain watching for the familiar "
    "flash that meant the channel was clear. By the time the sun finally broke through, "
    "the keeper had logged three hours of weather, wind speed, and visibility in the "
    "leather-bound book that sat on the desk beside the brass bell."
)


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
    max_diff = diff[finite].max().item() if nz else 0.0
    print(f"{tag}: DIVERGES  {tuple(a.shape)} | nonzero diffs: {nz} | nan mismatches: {nan_mismatch} | max abs diff: {max_diff:.6g}")
    return nan_mismatch == 0 and nz == 0


def row_norm_corr(a, b):
    """Pearson correlation of per-row L2 norms; orientation oracle."""
    na = a.float().pow(2).sum(-1).sqrt()
    nb = b.float().pow(2).sum(-1).sqrt()
    return float(np.corrcoef(na.numpy(), nb.numpy())[0, 1])


def main():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from distillkit.models import Qwen35SidecarForCausalLM

    torch.manual_seed(20260906)
    ok = True

    # ---- 1. gate 3: stock class vs sidecar subclass on the converted weights ----
    print("== Check 1: bit parity (stock class vs sidecar subclass) ==")
    print("Loading converted text decoder ...")
    stock_cls = Qwen3_5ForCausalLM.from_pretrained(CONVERTED_PATH, torch_dtype=torch.bfloat16)
    stock_cls.eval()
    print("Loading sidecar student ...")
    custom, info = Qwen35SidecarForCausalLM.from_pretrained(
        CONVERTED_PATH, torch_dtype=torch.bfloat16, output_loading_info=True
    )
    custom.eval()

    non_sidecar_missing = [k for k in info["missing_keys"] if ".sidecar." not in k]
    print(f"missing keys: {len(info['missing_keys'])} (sidecar: {len(info['missing_keys']) - len(non_sidecar_missing)})  "
          f"unexpected: {len(info['unexpected_keys'])}")
    assert not info["unexpected_keys"], f"unexpected keys: {info['unexpected_keys'][:5]}"
    assert not non_sidecar_missing, f"converted weights did not all load: {non_sidecar_missing[:5]}"

    sidecar = custom.model.layers[1].sidecar
    assert torch.count_nonzero(sidecar.W_side_proj.weight) == 0
    assert all(torch.count_nonzero(b.weight) == 0 for b in sidecar.gated_residual.branches)

    g = torch.Generator().manual_seed(7)
    raw = torch.randint(0, 256, (1, 256, 16, 90), dtype=torch.uint8, generator=g)
    scale = torch.full((1, 256, 16, 1), 0.001, dtype=torch.float16)
    raw.view(1, 256, 16, 5, 18)[..., :2] = (
        scale.view(torch.uint8).view(1, 256, 16, 1, 2).expand(1, 256, 16, 5, 2)
    )

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CONVERTED_PATH)
    ids = torch.tensor([tok(PASSAGE, add_special_tokens=False)["input_ids"]], dtype=torch.long)
    ids = ids[:, :256]
    print(f"tokenized passage: {ids.shape[1]} tokens")

    with torch.no_grad():
        out_s = stock_cls(ids, output_hidden_states=True)
        out_c = custom(ids, ngram_raw=raw[:, : ids.shape[1]], output_hidden_states=True)

    ok &= compare("logits", out_s.logits, out_c.logits)
    assert len(out_s.hidden_states) == len(out_c.hidden_states) == 33
    for i, (hs, hc) in enumerate(zip(out_s.hidden_states, out_c.hidden_states)):
        if not torch.equal(hs, hc):
            ok &= compare(f"hidden_states[{i}]", hs, hc)

    # ---- 2. orientation oracle vs student-stock ----
    print("\n== Check 2: row-norm correlation vs student-stock ==")
    print("Loading stock text decoder ...")
    ref = Qwen3_5ForCausalLM.from_pretrained(STOCK_PATH, torch_dtype=torch.bfloat16)
    ref.eval()

    probes = [
        ("model.embed_tokens.weight", "model.embed_tokens.weight"),
        ("model.layers.0.linear_attn.in_proj_qkv.weight", "model.layers.0.linear_attn.in_proj_qkv.weight"),
        ("model.layers.1.linear_attn.in_proj_a.weight", "model.layers.1.linear_attn.in_proj_a.weight"),
        ("model.layers.1.linear_attn.in_proj_b.weight", "model.layers.1.linear_attn.in_proj_b.weight"),
        ("model.layers.3.self_attn.q_proj.weight", "model.layers.3.self_attn.q_proj.weight"),
        ("model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.gate_proj.weight"),
    ]
    for key, ref_key in probes:
        a = custom.state_dict()[key]
        b = ref.state_dict()[ref_key]
        assert a.shape == b.shape, f"shape mismatch {key}: {a.shape} vs {b.shape}"
        r = row_norm_corr(a, b)
        flag = "OK " if r > 0.9 else "LOW"
        print(f"  [{flag}] corr({key}) = {r:.4f}")
        ok &= r > 0.9
    del ref

    # ---- 3. differential LM loss on the same real tokens ----
    print("\n== Check 3: differential LM loss (converted fine-tune vs stock) ==")
    with torch.no_grad():
        loss_conv = float(stock_cls(ids, labels=ids).loss)
        ref2 = Qwen3_5ForCausalLM.from_pretrained(STOCK_PATH, torch_dtype=torch.bfloat16)
        ref2.eval()
        loss_stock = float(ref2(ids, labels=ids).loss)
        del ref2
    print(f"  converted (Qwen3.8-4B fine-tune): {loss_conv:.4f}")
    print(f"  stock       (Qwen3.5-4B base):    {loss_stock:.4f}")
    ok &= np.isfinite(loss_conv) and loss_conv < 8.0
    ok &= np.isfinite(loss_stock) and loss_stock < 8.0

    print("\nCONVERTED-STUDENT VERIFICATION:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
