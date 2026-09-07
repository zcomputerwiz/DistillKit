"""Validate the converted 27B teacher against the official BF16 release.

The student conversion was validated three ways, but none of that transfers
automatically: the 27B has 64 layers instead of 32, an *untied* output head, and a
different linear-attention head ratio (16 key / 48 value, so 3 V-heads per K-head
where the student had 2). The grouped-to-tiled V reorder is exactly the kind of
convention that is silently wrong at a new ratio -- the checkpoint still loads, still
produces finite logits, and is quietly scrambled.

Rather than download 55.6 GB, this range-reads individual tensors from the official
repo over HTTP (a few hundred MB at most) and compares them to the converted file.

Values will not match bitwise: the local source is Q8_K_XL, so the converted weights
are dequantized Q8. The test is therefore cosine similarity per tensor, with a
deliberately blunt control -- a *shuffled* copy of the same tensor -- so "0.999" is
demonstrably better than what a scrambled ordering would score.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys

import numpy as np
import torch
from huggingface_hub import HfFileSystem, hf_hub_download
from safetensors import safe_open

REPO = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

# One per convention the converter applies, so a failure names the bug:
#   embed/lm_head  -> plain copy + untied-head detection
#   in_proj_qkv    -> the V-head grouped->tiled inverse permutation
#   in_proj_a/b    -> the same permutation on a different axis
#   conv1d         -> V-channel permutation plus the [C,1,K] reshape
#   out_proj       -> the permutation applied to COLUMNS, not rows
#   A_log          -> log(-stored) inverse
#   norms          -> the (w + 1) -> w offset, and the one norm exempt from it
PROBES = [
    "model.embed_tokens.weight",
    "lm_head.weight",
    "model.norm.weight",
    "model.layers.0.linear_attn.in_proj_qkv.weight",
    "model.layers.0.linear_attn.in_proj_a.weight",
    "model.layers.0.linear_attn.in_proj_b.weight",
    "model.layers.0.linear_attn.conv1d.weight",
    "model.layers.0.linear_attn.out_proj.weight",
    "model.layers.0.linear_attn.A_log",
    "model.layers.0.linear_attn.norm.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.3.self_attn.q_proj.weight",   # layer 3 is full_attention
    "model.layers.3.self_attn.o_proj.weight",
    "model.layers.3.self_attn.q_norm.weight",
    "model.layers.63.mlp.down_proj.weight",     # last layer: catches off-by-one
    "model.layers.63.self_attn.q_proj.weight",  # (63+1)%4==0, so 63 is FULL attention
    "model.layers.62.linear_attn.in_proj_qkv.weight",  # deepest linear-attention layer
    "model.layers.62.linear_attn.out_proj.weight",
]

_DTYPE = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}


MAX_ROWS = 256  # rows fetched per tensor; embed/lm_head are 248320 x 5120


def official_key(key: str) -> str:
    """Text-only key -> the VLM checkpoint's key.

    The official 27B release is Qwen3_5ForConditionalGeneration, so its decoder lives
    under `model.language_model.*`. `lm_head` stays top-level.
    """
    return key.replace("model.", "model.language_model.", 1) if key.startswith("model.") else key


def official_tensor(fs, weight_map, header_cache, key):
    """Fetch a tensor -- or its first MAX_ROWS rows -- by HTTP range.

    Rows are contiguous in safetensors, so a leading row slice is one range request.
    Comparing 256 x 5120 of the embedding is as diagnostic as all 248320 rows and
    costs 2.5 MB instead of 2.5 GB.
    """
    key = official_key(key)
    filename = weight_map[key]
    if filename not in header_cache:
        with fs.open(f"{REPO}@{REVISION}/{filename}", "rb", block_size=1, cache_type="none") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header_cache[filename] = (n, json.loads(fh.read(n)))
    n, header = header_cache[filename]
    entry = header[key]
    start, stop = entry["data_offsets"]
    shape = list(entry["shape"])
    dtype = _DTYPE[entry["dtype"]]
    itemsize = torch.empty(0, dtype=dtype).element_size()

    rows = shape[0]
    take = min(rows, MAX_ROWS) if len(shape) >= 2 else rows
    row_bytes = (stop - start) // rows if rows else 0
    length = take * row_bytes if len(shape) >= 2 else (stop - start)

    with fs.open(f"{REPO}@{REVISION}/{filename}", "rb", block_size=1, cache_type="none") as fh:
        fh.seek(8 + n + start)
        raw = fh.read(length)
    flat = torch.frombuffer(bytearray(raw), dtype=dtype)
    sliced = [take] + shape[1:] if len(shape) >= 2 else shape
    return flat.reshape(sliced).float().numpy(), shape, take


def cosine(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted", default="../teacher-hf")
    parser.add_argument("--report")
    args = parser.parse_args()

    idx = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json", revision=REVISION)))
    weight_map = idx["weight_map"]
    fs = HfFileSystem()
    header_cache: dict = {}

    local_path = f"{args.converted}/model.safetensors"
    rows, worst = [], 1.0
    with safe_open(local_path, framework="pt") as local:
        local_keys = set(local.keys())
        print(f"converted tensors: {len(local_keys)}")
        missing = [k for k in PROBES if k not in local_keys]
        if missing:
            print("MISSING from converted checkpoint:", missing)
            return 2

        print(f"\n{'tensor':52s} {'shape':>18s} {'cosine':>9s} {'shuffled':>9s}")
        for key in PROBES:
            got_full = local.get_tensor(key).float().numpy()
            ref, shape, take = official_tensor(fs, weight_map, header_cache, key)
            if tuple(got_full.shape) != tuple(shape):
                print(f"  {key:50s} SHAPE MISMATCH converted {got_full.shape} vs official {tuple(shape)}")
                return 2
            got = got_full[:take] if len(shape) >= 2 else got_full
            c = cosine(ref, got)
            control = cosine(ref, np.random.default_rng(0).permutation(got.reshape(-1)))
            worst = min(worst, c)
            rows.append({"key": key, "cosine": c, "shuffled_control": control,
                         "shape": list(shape)})
            print(f"  {key:50s} {str(tuple(shape)):>18s} {c:9.5f} {control:9.5f}")

    print(f"\nworst cosine {worst:.5f} over {len(rows)} tensors")
    passed = worst > 0.99
    print("TEACHER CONVERSION:", "PASS" if passed else "FAIL",
          "-- dequantized Q8 matches the official BF16 weights" if passed
          else "-- a conversion convention is wrong for this geometry")
    if args.report:
        json.dump({"repo": REPO, "revision": REVISION, "worst_cosine": worst,
                   "passed": bool(passed), "tensors": rows},
                  open(args.report, "w"), indent=2)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
