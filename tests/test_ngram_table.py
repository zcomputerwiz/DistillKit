"""Gates on the IQ4_NL n-gram table provider.

Two tiers. The dequant tests need only the ``gguf`` package and run anywhere.
The layout tests need the local 49.8 GB shard and are skipped without it -- but
when it is present they are the canary for "did the table's row order survive":
if a re-download or re-quant ever transposes or permutes the tensor, they fail
before a single training step is wasted on garbage rows.
"""

import os

import numpy as np
import pytest
import torch

from distillkit.ngram_hash import FLASH_NEXT_NGRAM_CONFIG, NGramHasher
from distillkit.ngram_table import (
    FLASH_NEXT_TABLE,
    IQ4NL_BLOCK,
    IQ4NL_KVALUES,
    IQ4NL_TYPE_SIZE,
    GGUFNGramTable,
    IQ4NLDequant,
    dequantize_iq4nl_rows,
    read_gguf_ple_metadata,
)

gguf = pytest.importorskip("gguf", reason="gguf package not installed")

GGUF_DIR = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "huggingface",
    "hub",
    "models--unsloth--Qwen3.8-Flash-Next-GGUF",
    "snapshots",
    "38bb39ee97821de2c9009abb7e93950eec396e66",
    "UD-IQ4_XS",
)
SHARD_1 = os.path.join(GGUF_DIR, "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf")
SHARD_2 = os.path.join(GGUF_DIR, "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf")

needs_shard_2 = pytest.mark.skipif(
    not os.path.exists(SHARD_2), reason=f"local GGUF shard not present: {SHARD_2}"
)
needs_shard_1 = pytest.mark.skipif(
    not os.path.exists(SHARD_1), reason=f"local GGUF shard not present: {SHARD_1}"
)


# --- geometry (no files needed) ----------------------------------------------


def test_geometry_matches_hasher():
    hasher = NGramHasher(FLASH_NEXT_NGRAM_CONFIG)
    assert FLASH_NEXT_TABLE.n_rows == hasher.padded_vocab_size == 320_001_536
    assert FLASH_NEXT_TABLE.n_real_rows == hasher.total_vocab_size == 320_001_446
    assert FLASH_NEXT_TABLE.head_dim == FLASH_NEXT_NGRAM_CONFIG.head_dim == 160
    assert FLASH_NEXT_TABLE.blocks_per_row == 5
    assert FLASH_NEXT_TABLE.bytes_per_row == 90
    assert FLASH_NEXT_TABLE.n_bytes == 28_800_138_240


def test_format_constants_match_gguf_package():
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
    from gguf.quants import IQ4_NL

    assert GGML_QUANT_SIZES[GGMLQuantizationType.IQ4_NL] == (IQ4NL_BLOCK, IQ4NL_TYPE_SIZE)
    assert tuple(IQ4_NL.kvalues) == IQ4NL_KVALUES


# --- dequant parity (no files needed) ----------------------------------------


def _random_rows(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=(n, 5, 18), dtype=np.uint8)
    # Keep the fp16 scale finite so the reference and we agree on non-NaN values.
    scales = (rng.standard_normal((n, 5)) * 0.01).astype(np.float16)
    raw[..., :2] = scales.view(np.uint8).reshape(n, 5, 2)
    return raw.reshape(n, 90)


def test_numpy_dequant_matches_gguf_reference():
    from gguf.quants import IQ4_NL

    raw = _random_rows(257, seed=1)
    ours = dequantize_iq4nl_rows(raw, out_dtype=np.float32)
    ref = IQ4_NL.dequantize_blocks(raw.reshape(-1, 18)).reshape(257, 160)
    assert ours.shape == (257, 160)
    assert np.array_equal(ours, ref), "numpy dequant diverged from gguf.quants.IQ4_NL"


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_torch_dequant_matches_numpy(device):
    raw = _random_rows(512, seed=2)
    expected = dequantize_iq4nl_rows(raw, out_dtype=np.float32)
    module = IQ4NLDequant(out_dtype=torch.float32).to(device)
    out = module(torch.from_numpy(raw).to(device))
    assert out.shape == (512, 160)
    assert torch.equal(out.cpu(), torch.from_numpy(expected)), "torch dequant != numpy dequant"


def test_torch_dequant_preserves_leading_dims():
    raw = _random_rows(2 * 3 * 16, seed=3).reshape(2, 3, 16, 90)
    out = IQ4NLDequant(out_dtype=torch.float32)(torch.from_numpy(raw))
    assert out.shape == (2, 3, 16, 160)
    flat = dequantize_iq4nl_rows(raw.reshape(-1, 90)).reshape(2, 3, 16, 160)
    assert torch.equal(out, torch.from_numpy(flat))


def test_nibble_order_low_then_high():
    """Byte j low nibble -> element j, high nibble -> element j+16. A swap here yields
    plausible-looking wrong embeddings that no downstream metric would catch."""
    raw = np.zeros((1, 90), dtype=np.uint8)
    raw[0, 0:2] = np.array([1.0], dtype=np.float16).view(np.uint8)  # scale 1.0, block 0
    raw[0, 2] = 0xF0  # byte 0 of block 0: low nibble 0 -> kvalues[0], high nibble 15 -> kvalues[15]
    out = dequantize_iq4nl_rows(raw)[0]
    assert out[0] == IQ4NL_KVALUES[0] == -127
    assert out[16] == IQ4NL_KVALUES[15] == 113
    # every other element in the block indexes nibble 0 -> -127 as well
    assert out[1] == -127 and out[17] == -127


def test_torch_dequant_rejects_non_uint8():
    with pytest.raises(TypeError):
        IQ4NLDequant()(torch.zeros(1, 90, dtype=torch.int64))


# --- hasher / table contract -------------------------------------------------


def test_hasher_never_emits_padding_rows():
    """Rows >= total_vocab_size are zero padding. The remainder+offset construction
    bounds every head's rows inside its own prime-sized band, so they are unreachable --
    assert it, because a config drift here would silently feed zero vectors."""
    hasher = NGramHasher(FLASH_NEXT_NGRAM_CONFIG)
    generator = torch.Generator().manual_seed(11)
    ids = torch.randint(0, FLASH_NEXT_NGRAM_CONFIG.vocab_size, (4, 2048), generator=generator)
    rows = hasher.row_indices(ids)
    assert rows.max().item() < FLASH_NEXT_TABLE.n_real_rows
    # Tight: the last head's band ends exactly at n_real_rows.
    last = hasher.head_offsets[-1].item() + hasher.head_vocab_sizes[-1].item()
    assert last == FLASH_NEXT_TABLE.n_real_rows


# --- against the real shard --------------------------------------------------


@pytest.fixture(scope="module")
def table():
    if not os.path.exists(SHARD_2):
        pytest.skip("shard 2 absent")
    return GGUFNGramTable(SHARD_2)


@needs_shard_2
def test_table_locates_tensor(table):
    assert table.n_bytes == 28_800_138_240
    assert table.raw.shape == (320_001_536, 90)
    assert table.raw.dtype == np.uint8
    assert not table.resident


@needs_shard_2
def test_padding_boundary_is_exactly_total_vocab_size(table):
    """The layout canary. Rows from total_vocab_size onward must be exactly zero and the
    row just before must not be. This detects a misplaced padding boundary, but
    does not rule out permutations within the real rows; the BF16 check does that."""
    n_real = FLASH_NEXT_TABLE.n_real_rows
    pad = table.gather(np.arange(n_real, FLASH_NEXT_TABLE.n_rows))
    assert pad.shape == (90, 160)
    assert np.all(pad == 0.0)
    last_real = table.row(n_real - 1)
    assert np.linalg.norm(last_real) > 0.01
    assert np.count_nonzero(last_real) == 160


@needs_shard_2
def test_head_partition_signature(table):
    """Bigram heads (0-7) and trigram heads (8-15) have distinct per-row-norm spread;
    the split lands exactly at head 8 only if row order preserves the head partition."""
    hasher = NGramHasher(FLASH_NEXT_NGRAM_CONFIG)
    rng = np.random.default_rng(0)
    spreads = []
    for head in range(16):
        lo = hasher.head_offsets[head].item()
        hi = lo + hasher.head_vocab_sizes[head].item()
        rows = rng.integers(lo, hi, size=400)
        norms = np.linalg.norm(table.gather(rows), axis=-1)
        spreads.append(norms.std())
    bigram, trigram = np.mean(spreads[:8]), np.mean(spreads[8:])
    assert bigram > 0.010, f"bigram head spread {bigram:.4f} too small"
    assert trigram < 0.0085, f"trigram head spread {trigram:.4f} too large"
    assert min(spreads[:8]) > max(spreads[8:]), "head groups overlap -- partition scrambled?"


@needs_shard_2
def test_rows_are_plausible_embeddings_not_noise(table):
    rng = np.random.default_rng(1)
    rows = table.gather(rng.integers(0, FLASH_NEXT_TABLE.n_real_rows, size=2048))
    assert np.isfinite(rows).all()
    assert (np.count_nonzero(rows, axis=-1) == 160).all(), "dead rows present"
    per_row_l2 = np.linalg.norm(rows, axis=-1)
    assert 0.05 < per_row_l2.mean() < 0.2
    assert abs(rows.mean()) < 1e-3


@needs_shard_2
def test_gather_raw_and_gather_agree_with_gpu_path(table):
    hasher = NGramHasher(FLASH_NEXT_NGRAM_CONFIG)
    ids = torch.tensor([[248044, 17, 3402, 9982, 5, 248000]], dtype=torch.long)
    rows = hasher.row_indices(ids)  # (1, 6, 16)
    raw = table.gather_raw(rows)
    assert raw.shape == (1, 6, 16, 90)
    host = table.gather(rows)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = IQ4NLDequant(out_dtype=torch.float32).to(dev)(torch.from_numpy(raw).to(dev)).cpu()
    assert torch.equal(gpu, torch.from_numpy(host))
    assert gpu.shape == (1, 6, 16, 160)


@needs_shard_1
def test_gguf_kv_metadata_matches_hasher_derivation():
    """Third independent oracle for the hash constants: the converter wrote the
    reference module's buffers verbatim into the GGUF KV block."""
    meta = read_gguf_ple_metadata(SHARD_1)
    hasher = NGramHasher(FLASH_NEXT_NGRAM_CONFIG)
    assert meta["qwen4exp.ple.layer_multipliers"] == hasher.layer_multipliers.tolist()
    assert meta["qwen4exp.ple.head_offsets"] == hasher.head_offsets.tolist()
    assert meta["qwen4exp.ple.head_vocab_sizes"] == hasher.head_vocab_sizes.tolist()
    assert meta["qwen4exp.ple.ngram_size"] == [FLASH_NEXT_NGRAM_CONFIG.ngram_size]
    assert meta["qwen4exp.ple.heads_per_ngram"] == [FLASH_NEXT_NGRAM_CONFIG.heads_per_ngram]
    assert meta["qwen4exp.ple.eos_token_id"] == [FLASH_NEXT_NGRAM_CONFIG.eos_token_id]
    assert meta["qwen4exp.embedding_length_per_layer_input"] == [FLASH_NEXT_NGRAM_CONFIG.head_dim]
    # GGUF's ple.layers is ZERO-indexed (config.json's ple_layer_ids=[2] is one-indexed);
    # both name decoder layer 1.
    assert meta["qwen4exp.ple.layers"] == [1]
