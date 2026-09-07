"""Close acceptance criterion 2 for the GGUF table: do its rows match the reference?

Structural checks (zero-pad boundary, head-partition signature, KV metadata) already
establish the GGUF row order is not transposed or permuted. What remains is whether
GGUF row i *is* reference row i element-wise, up to IQ4_NL quantization.

The official ``Qwen/Qwen3.8-Flash-Next`` release stores the table as BF16 in 128
safetensors shards of ``[2500012, 160]`` (``ngram_embedding.shard_k.weight``), so a
row is 320 contiguous bytes at a computable offset. We fetch exactly those bytes with
HTTP range requests -- a few KB total -- rather than 102 GB of weights.

For each of several token triples: hash -> 16 global rows -> reference rows (BF16 over
HTTP) vs GGUF rows (IQ4_NL local). Report cosine and relative L2. As a control, also
score each GGUF row against a *different* reference row; a real match should sit near
cosine 1.0 and the control near 0.0. If they do not separate, the layout is wrong.
"""

import argparse
import functools
import json
import os
import struct
import sys

import numpy as np
import torch
from huggingface_hub import HfFileSystem, hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distillkit.ngram_hash import FLASH_NEXT_NGRAM_CONFIG, NGramHasher  # noqa: E402
from distillkit.ngram_table import GGUFNGramTable  # noqa: E402

REPO = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
PREFIX = "model.language_model.layers.1.ple.ple_embedding."
ROWS_PER_SHARD = 2_500_012
BYTES_PER_ROW = 160 * 2  # bf16

GGUF_SHARD_2 = os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "models--unsloth--Qwen3.8-Flash-Next-GGUF", "snapshots",
    "38bb39ee97821de2c9009abb7e93950eec396e66", "UD-IQ4_XS",
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf",
)

fs = HfFileSystem()
_header_cache: dict[str, tuple[int, dict]] = {}


def shard_header(filename: str) -> tuple[int, dict]:
    """(header_len, header_json) for one safetensors file, via two range reads."""
    if filename in _header_cache:
        return _header_cache[filename]
    with fs.open(f"{REPO}@{REVISION}/{filename}", "rb", block_size=1, cache_type="none") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        if not 2 <= n <= 64 * 1024 * 1024:
            raise ValueError(f"Invalid safetensors header length: {n}")
        hdr = json.loads(fh.read(n))
    _header_cache[filename] = (n, hdr)
    return _header_cache[filename]


@functools.lru_cache(maxsize=256)
def read_range(filename: str, start: int, length: int) -> bytes:
    # block_size=0 selects HfFileSystemStreamFile, which cannot seek.
    # A nonzero block size plus no cache requests exactly the selected bytes.
    with fs.open(f"{REPO}@{REVISION}/{filename}", "rb", block_size=1, cache_type="none") as fh:
        fh.seek(start)
        data = fh.read(length)
    if len(data) != length:
        raise ValueError(f"Short range read: {filename}: expected {length}, got {len(data)}")
    return data


def reference_row(weight_map: dict, global_row: int) -> np.ndarray:
    k, local = divmod(global_row, ROWS_PER_SHARD)
    key = f"{PREFIX}ngram_embedding.shard_{k}.weight"
    filename = weight_map[key]
    n, hdr = shard_header(filename)
    entry = hdr[key]
    assert entry["dtype"] == "BF16", entry
    assert entry["shape"] == [ROWS_PER_SHARD, 160], entry
    start = 8 + n + entry["data_offsets"][0] + local * BYTES_PER_ROW
    raw = read_range(filename, start, BYTES_PER_ROW)
    assert len(raw) == BYTES_PER_ROW, len(raw)
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float().numpy()


def reference_buffers(weight_map: dict) -> dict[str, list[int]]:
    """The reference module's int64 buffers, straight from safetensors -- oracle #4."""
    out = {}
    for name in ("ngram_heads_offsets", "ngram_heads_vocab_sizes", "layer_multipliers"):
        key = PREFIX + name
        filename = weight_map[key]
        n, hdr = shard_header(filename)
        entry = hdr[key]
        a, b = entry["data_offsets"]
        raw = read_range(filename, 8 + n + a, b - a)
        out[name] = np.frombuffer(raw, dtype=np.int64).tolist()
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", default=GGUF_SHARD_2)
    parser.add_argument("--report", help="Write reproducible JSON verification results")
    args = parser.parse_args()
    idx_path = hf_hub_download(REPO, "model.safetensors.index.json", revision=REVISION)
    with open(idx_path, encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]
    print("Reference:", REPO, "revision", REVISION)

    hasher = NGramHasher(FLASH_NEXT_NGRAM_CONFIG)
    print("== oracle #4: reference buffers in safetensors ==")
    bufs = reference_buffers(weight_map)
    ok = (
        bufs["ngram_heads_offsets"] == hasher.head_offsets.tolist()
        and bufs["ngram_heads_vocab_sizes"] == hasher.head_vocab_sizes.tolist()
        and bufs["layer_multipliers"] == hasher.layer_multipliers.tolist()
    )
    print("  buffers match hasher derivation:", ok)
    if not ok:
        print("  ", bufs)
        return 2

    table = GGUFNGramTable(args.gguf)
    print("==", table)

    triples = [
        [248044, 17, 3402],
        [1, 2, 3],
        [9982, 5, 248000],
        [100000, 200000, 248319],
        [42, 42, 42],
    ]
    ids = torch.tensor(triples, dtype=torch.long)
    rows = hasher.row_indices(ids)[:, -1, :]  # last position sees the full trigram context
    all_cos, all_rel, ctrl_cos = [], [], []
    print("\n== per-head comparison (16 rows per triple) ==")
    for t, triple in enumerate(triples):
        for head in range(16):
            r = int(rows[t, head])
            ref = reference_row(weight_map, r)
            got = table.row(r)
            c = cosine(ref, got)
            rel = float(np.linalg.norm(ref - got) / (np.linalg.norm(ref) + 1e-12))
            # control: same GGUF row scored against the reference row for a *different* head
            other = int(rows[t, (head + 5) % 16])
            ctrl = cosine(reference_row(weight_map, other), got)
            all_cos.append(c); all_rel.append(rel); ctrl_cos.append(ctrl)
            if head < 2 or head in (8, 15):
                print(f"  triple {t} head {head:2d} row {r:>11d}: cos {c:.4f} rel_l2 {rel:.3f} | control cos {ctrl:+.3f}")

    all_cos, all_rel, ctrl_cos = map(np.array, (all_cos, all_rel, ctrl_cos))
    print("\n== summary over", len(all_cos), "rows ==")
    print(f"  match   cosine: mean {all_cos.mean():.4f} min {all_cos.min():.4f}")
    print(f"  match   rel L2: mean {all_rel.mean():.3f} max {all_rel.max():.3f}")
    print(f"  control cosine: mean {ctrl_cos.mean():+.4f} max {np.abs(ctrl_cos).max():.4f}")

    passed = all_cos.min() > 0.90 and np.abs(ctrl_cos).max() < 0.5
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump({
                "repo": REPO, "revision": REVISION, "gguf": args.gguf,
                "tensor_type": "IQ4_NL", "table_bytes": table.n_bytes,
                "reference_buffers_match": ok, "token_triples": triples,
                "row_ids": rows.tolist(), "cosines": all_cos.tolist(),
                "relative_l2": all_rel.tolist(), "control_cosines": ctrl_cos.tolist(),
                "passed": bool(passed),
            }, fh, indent=2)
            fh.write("\n")
    print("\nCRITERION 2:", "PASS" if passed else "FAIL",
          "-- GGUF row i is reference row i (up to IQ4_NL quantization)" if passed
          else "-- rows do not match the reference; do NOT use this table")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
