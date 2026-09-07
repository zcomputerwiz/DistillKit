"""Would the 27B teacher GGUF convert cleanly? Check config + tensor mapping only.

No 55 GB write: this derives the config from KV metadata, maps every tensor name, and
dequantizes a small sample of each distinct ggml type, so a geometry or type problem
surfaces in seconds rather than after a long conversion.
"""
import os, sys
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gguf import GGUFReader
from distillkit.convert_gguf_student import (
    _decode_tensor,
    derive_text_config,
    map_gguf_tensor,
    read_kv,
)

GGUF = r"C:\Users\Owner\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\4ca720788d1e01f1bff70c033e0d0028fd02e502\Qwen3.8-27B-UD-Q8_K_XL.gguf"
r = GGUFReader(GGUF, mode="r")

kv = read_kv(r)
embd = next(t for t in r.tensors if t.name == "token_embd.weight")
vocab_size = int(embd.shape[-1])  # GGUF ne is reversed; rows are the vocabulary
has_head = any(t.name == "output.weight" for t in r.tensors)
cfg = derive_text_config(kv, vocab_size=vocab_size, tie_word_embeddings=not has_head)
print(f"  output.weight present = {has_head} -> tie_word_embeddings = {not has_head}")
for k in ("num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads",
          "num_key_value_heads", "head_dim", "vocab_size", "mtp_num_hidden_layers",
          "linear_num_key_heads", "linear_num_value_heads", "tie_word_embeddings"):
    print(f"  {k} = {cfg.get(k)}")
print(f"  layer_types: {Counter(cfg['layer_types']).most_common()}")

mapped, skipped, unmapped = 0, [], []
by_type = {}
for t in r.tensors:
    m = map_gguf_tensor(t.name, cfg["num_hidden_layers"])
    if m is None:
        (skipped if t.name.startswith(f"blk.{cfg['num_hidden_layers']}") else unmapped).append(t.name)
    else:
        mapped += 1
    by_type.setdefault(t.tensor_type.name, t)

print(f"\nmapped {mapped}, mtp-skipped {len(skipped)}, UNMAPPED {len(unmapped)}")
if unmapped:
    print("  unmapped:", unmapped[:10])

print("\ndequant probe, one tensor per ggml type:")
ok = True
for tname, t in by_type.items():
    try:
        a = _decode_tensor(r, t)
        finite = np.isfinite(a).all()
        print(f"  {tname:6s} {t.name[:42]:42s} -> {a.shape} {a.dtype} finite={finite} "
              f"absmax={np.abs(a).max():.4f}")
        ok &= bool(finite)
    except Exception as e:
        ok = False
        print(f"  {tname:6s} {t.name[:42]:42s} -> FAILED {type(e).__name__}: {e}")

print("\nDRY RUN:", "PASS" if (ok and not unmapped) else "FAIL")

# --- key-set pre-flight: catch mismatches in seconds, not after a 10-minute decode ---
from distillkit.convert_gguf_student import expected_key_set
produced = set()
for t in r.tensors:
    m = map_gguf_tensor(t.name, cfg["num_hidden_layers"])
    if m is not None:
        produced.add(m[0])
wanted = expected_key_set(cfg)
missing, extra = wanted - produced, produced - wanted
print(f"\nkey set: produced {len(produced)}, wanted {len(wanted)}")
print(f"  missing={sorted(missing)[:6]}")
print(f"  extra  ={sorted(extra)[:6]}")
print("KEY SET:", "PASS" if not (missing or extra) else "FAIL")
