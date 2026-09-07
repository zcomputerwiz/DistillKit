"""Row provider for the frozen Qwen3.8-Flash-Next n-gram table, served from a GGUF.

``distillkit.ngram_hash.NGramHasher`` turns token ids into ``[B, T, 16]`` global row
indices and deliberately owns no table. This module is the other half: it hands back
the 160-dim rows those indices name.

Why a GGUF, and why IQ4_NL specifically
---------------------------------------
The official ``Qwen/Qwen3.8-Flash-Next`` release stores the table as BF16 across 128
safetensors shards -- 102.4 GB. That does not fit in 128 GB of host RAM next to
ZeRO-2 optimizer offload for a 4B student. The Unsloth ``UD-IQ4_XS`` GGUF carries the
same tensor as ``per_layer_token_embd.weight`` at IQ4_NL (the dynamic scheme bumped
it *up* from the directory's nominal IQ4_XS), 4.5 bits/weight, 28.80 GB.

IQ4_NL is the friendliest possible quant to random-access: flat 32-element blocks, one
fp16 scale plus 16 bytes of packed nibbles per block, no superblock, no hierarchical
scales. A 160-element row is exactly 5 blocks = exactly 90 bytes, byte-aligned, never
straddling. ``raw[i]`` *is* row ``i``. That the GGUF row order matches the reference
was checked structurally (zero-padding rows begin at exactly
``total_vocab_size``; the bigram/trigram head statistics split at exactly head 8) and
is supported by ``tests/test_ngram_table.py``; sampled value agreement with the BF16
reference is checked by ``scratch/ngram_reference_check.py``.

Residency is the whole game
---------------------------
Dequant is cheap. Page faults are not. Measured on this box: the gather+dequant of a
full 8x4096 batch (524,288 rows) is ~0.6 s single-threaded when the table is RAM-
resident, and ~95 s when served cold from mmap -- one 4 KB fault per 90-byte row on
Windows' single-threaded fault path. Nothing else about the design matters if the
table is not resident. Hence :meth:`GGUFNGramTable.prefault` and, when the RAM budget
allows it, :meth:`GGUFNGramTable.load_resident`, which trades 28.8 GB of committed
process memory for immunity to page-cache eviction under ZeRO-2 pressure.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

LOG = logging.getLogger(__name__)

__all__ = [
    "IQ4NL_KVALUES",
    "IQ4NL_BLOCK",
    "IQ4NL_TYPE_SIZE",
    "NGramTableSpec",
    "FLASH_NEXT_TABLE",
    "dequantize_iq4nl_rows",
    "IQ4NLDequant",
    "GGUFNGramTable",
    "read_gguf_ple_metadata",
]

# --- IQ4_NL format constants (ggml) ------------------------------------------
# From gguf.quants.IQ4_NL / ggml-quants.c. A block is 32 elements: 2 bytes fp16 scale
# then 16 bytes of nibbles. Byte j's LOW nibble is element j, its HIGH nibble is
# element j+16. Values index a fixed non-linear 16-entry codebook.

IQ4NL_KVALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)
IQ4NL_BLOCK = 32
IQ4NL_TYPE_SIZE = 18

_KVALUES_NP = np.array(IQ4NL_KVALUES, dtype=np.int8)


@dataclass(frozen=True)
class NGramTableSpec:
    """Geometry of the table. Everything here is derivable from the Flash-Next config."""

    n_rows: int = 320_001_536  # padded_vocab_size
    n_real_rows: int = 320_001_446  # total_vocab_size; rows past this are zero padding
    head_dim: int = 160

    def __post_init__(self) -> None:
        if self.n_rows <= 0 or not 0 < self.n_real_rows <= self.n_rows:
            raise ValueError("table requires 0 < n_real_rows <= n_rows")
        if self.head_dim <= 0 or self.head_dim % IQ4NL_BLOCK:
            raise ValueError("head_dim must be a positive multiple of 32")

    @property
    def blocks_per_row(self) -> int:
        return self.head_dim // IQ4NL_BLOCK

    @property
    def bytes_per_row(self) -> int:
        return self.blocks_per_row * IQ4NL_TYPE_SIZE

    @property
    def n_bytes(self) -> int:
        return self.n_rows * self.bytes_per_row


FLASH_NEXT_TABLE = NGramTableSpec()


# --- dequantization ----------------------------------------------------------


def dequantize_iq4nl_rows(raw: np.ndarray, out_dtype=np.float32) -> np.ndarray:
    """``(..., 90) uint8`` -> ``(..., 160) out_dtype``.

    Vectorized port of ``gguf.quants.IQ4_NL.dequantize_blocks`` specialised to whole
    rows. Bit-identical to the reference in float32 (fp16 scale x small int is exact).
    """
    raw = np.asarray(raw)
    if raw.dtype != np.uint8:
        raise TypeError(f"expected uint8 rows, got {raw.dtype}")
    if raw.ndim == 0 or raw.shape[-1] == 0 or raw.shape[-1] % IQ4NL_TYPE_SIZE:
        raise ValueError("rows must contain a positive whole number of 18-byte IQ4_NL blocks")
    raw = np.ascontiguousarray(raw)
    lead = raw.shape[:-1]
    n_blocks_per_row = raw.shape[-1] // IQ4NL_TYPE_SIZE
    blocks = raw.reshape(-1, n_blocks_per_row, IQ4NL_TYPE_SIZE)

    scale = blocks[..., :2].copy().view(np.float16).astype(np.float32)  # (n, 5, 1)
    qs = blocks[..., 2:]  # (n, 5, 16)
    lo = qs & np.uint8(0x0F)
    hi = qs >> np.uint8(4)
    idx = np.concatenate([lo, hi], axis=-1)  # (n, 5, 32): elements 0..15 then 16..31
    values = _KVALUES_NP[idx].astype(np.float32)
    out = (scale * values).reshape(*lead, n_blocks_per_row * IQ4NL_BLOCK)
    return out.astype(out_dtype, copy=False)


class IQ4NLDequant(nn.Module):
    """Torch dequant, so the 90-byte rows can cross PCIe and unpack on the GPU.

    Shipping raw bytes instead of dequantized floats cuts host->device traffic from
    168-335 MB per 8x4096 batch to 47 MB, and moves the unpack off the CPU entirely.
    Bit-identical to :func:`dequantize_iq4nl_rows` in float32.
    """

    kvalues: torch.Tensor

    def __init__(self, out_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.out_dtype = out_dtype
        self.register_buffer(
            "kvalues", torch.tensor(IQ4NL_KVALUES, dtype=torch.float32), persistent=False
        )

    def forward(self, raw_u8: torch.Tensor) -> torch.Tensor:
        if raw_u8.dtype != torch.uint8:
            raise TypeError(f"expected uint8 rows, got {raw_u8.dtype}")
        if raw_u8.ndim == 0 or raw_u8.shape[-1] == 0 or raw_u8.shape[-1] % IQ4NL_TYPE_SIZE:
            raise ValueError("rows must contain a positive whole number of 18-byte IQ4_NL blocks")
        lead = raw_u8.shape[:-1]
        n_blocks_per_row = raw_u8.shape[-1] // IQ4NL_TYPE_SIZE
        blocks = raw_u8.reshape(-1, n_blocks_per_row, IQ4NL_TYPE_SIZE)

        scale = blocks[..., :2].contiguous().view(torch.float16).float()  # (n, 5, 1)
        qs = blocks[..., 2:]
        lo = (qs & 0x0F).long()
        hi = (qs >> 4).long()
        idx = torch.cat([lo, hi], dim=-1)  # (n, 5, 32)
        values = self.kvalues[idx]
        out = (scale * values).reshape(*lead, n_blocks_per_row * IQ4NL_BLOCK)
        return out.to(self.out_dtype)


# --- GGUF access -------------------------------------------------------------


def _locate_tensor(gguf_path: str, tensor_name: str) -> tuple[int, tuple[int, ...], int]:
    """Return ``(data_offset, raw_byte_shape, n_bytes)`` for a tensor, reading only the header."""
    from gguf import GGUFReader, GGMLQuantizationType  # local import: optional dependency

    reader = GGUFReader(gguf_path, mode="r")
    if reader.byte_order != "I":
        raise ValueError("byte-swapped GGUF tables are unsupported")
    for tensor in reader.tensors:
        if tensor.name == tensor_name:
            # Q4_0 has the SAME 18 bytes per 32 elements but a different codebook.
            # Shape/byte-count validation alone would silently decode it as IQ4_NL.
            if tensor.tensor_type != GGMLQuantizationType.IQ4_NL:
                raise ValueError(f"{tensor_name} must be IQ4_NL, got {tensor.tensor_type.name}")
            return int(tensor.data_offset), tuple(int(s) for s in tensor.data.shape), int(tensor.n_bytes)
    names = ", ".join(t.name for t in reader.tensors[:8])
    raise KeyError(f"{tensor_name!r} not in {gguf_path} (first tensors: {names} ...)")


def read_gguf_ple_metadata(gguf_path: str) -> dict[str, list[int]]:
    """The ``qwen4exp.ple.*`` KV block. Shard 1 of the Unsloth split carries it alone.

    These are the reference's ``head_offsets`` / ``head_vocab_sizes`` /
    ``layer_multipliers`` written verbatim by the converter -- an independent oracle
    for the values ``NGramHasher`` derives from the transformers code.
    """
    from gguf import GGUFReader

    reader = GGUFReader(gguf_path, mode="r")
    out: dict[str, list[int]] = {}
    for key, field in reader.fields.items():
        if not (key.startswith("qwen4exp.ple.") or key == "qwen4exp.embedding_length_per_layer_input"):
            continue
        values = []
        for i in field.data:
            part = field.parts[i].tolist()
            values.extend(part if isinstance(part, list) else [part])
        out[key] = values
    return out


class GGUFNGramTable:
    """Byte-addressable view of ``per_layer_token_embd.weight`` inside a GGUF shard.

    Holds the table as either a read-only ``np.memmap`` (default; relies on the OS
    page cache) or, after :meth:`load_resident`, an anonymous in-process array.
    Never a ``torch.nn.Parameter``, never a buffer, never in any state dict.
    """

    def __init__(
        self,
        gguf_path: str,
        tensor_name: str = "per_layer_token_embd.weight",
        spec: NGramTableSpec = FLASH_NEXT_TABLE,
    ) -> None:
        self.gguf_path = gguf_path
        self.tensor_name = tensor_name
        self.spec = spec

        data_offset, raw_shape, n_bytes = _locate_tensor(gguf_path, tensor_name)
        expected = (spec.n_rows, spec.bytes_per_row)
        if raw_shape != expected:
            raise ValueError(
                f"{tensor_name} raw byte shape {raw_shape} != expected {expected}; "
                "either the quant type is not IQ4_NL or this is not the Flash-Next table"
            )
        if n_bytes != spec.n_bytes:
            raise ValueError(f"{tensor_name} is {n_bytes} bytes, expected {spec.n_bytes}")

        self.data_offset = data_offset
        self.n_bytes = n_bytes
        self.raw: np.ndarray = np.memmap(
            gguf_path, dtype=np.uint8, mode="r", offset=data_offset, shape=expected
        )
        self.resident = False

    # -- residency ------------------------------------------------------------

    def prefault(self, chunk_bytes: int = 256 << 20) -> float:
        """Touch every page sequentially so the OS caches the whole table.

        Sequential read of 28.8 GB on NVMe is a few minutes; random faults during
        training are ~150x slower per byte. Call once, from the main process, before
        dataloader workers fork -- the page cache is shared, the fault cost is not.
        Returns wall-clock seconds.
        """
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        if self.resident:
            return 0.0
        flat = self.raw.reshape(-1)
        t0 = time.perf_counter()
        sink = 0
        for start in range(0, flat.shape[0], chunk_bytes):
            sink += int(flat[start : start + chunk_bytes][::4096].sum())  # one touch per page
        elapsed = time.perf_counter() - t0
        LOG.info("prefaulted %.2f GB in %.1fs (checksum %d)", self.n_bytes / 1e9, elapsed, sink)
        return elapsed

    def load_resident(self) -> float:
        """Copy the table into anonymous memory (28.8 GB) instead of trusting the page cache.

        Windows evicts page cache before it swaps, so under ZeRO-2 offload pressure the
        memmap can silently go cold mid-run. This removes that failure mode at the cost
        of committed RAM. Returns wall-clock seconds.
        """
        if self.resident:
            return 0.0
        t0 = time.perf_counter()
        arr = np.fromfile(
            self.gguf_path, dtype=np.uint8, count=self.n_bytes, offset=self.data_offset
        )
        self.raw = arr.reshape(self.spec.n_rows, self.spec.bytes_per_row)
        self.resident = True
        elapsed = time.perf_counter() - t0
        LOG.info("loaded %.2f GB resident in %.1fs", self.n_bytes / 1e9, elapsed)
        return elapsed

    # -- access ---------------------------------------------------------------

    def gather_raw(self, rows) -> np.ndarray:
        """``rows: (...) int`` -> ``(..., 90) uint8``. The cheap host-side half of a lookup."""
        if isinstance(rows, torch.Tensor):
            rows = rows.detach().cpu().numpy()
        rows = np.asarray(rows)
        if rows.dtype.kind not in "iu":
            raise TypeError("row indices must be integers")
        if rows.size and (np.any(rows < 0) or np.any(rows >= self.spec.n_rows)):
            raise IndexError("row index outside table")
        rows = rows.astype(np.int64, copy=False)
        return self.raw[rows.reshape(-1)].reshape(*rows.shape, self.spec.bytes_per_row)

    def gather(self, rows, out_dtype=np.float32) -> np.ndarray:
        """``rows: (...) int`` -> ``(..., 160) out_dtype``, dequantized on the host."""
        return dequantize_iq4nl_rows(self.gather_raw(rows), out_dtype=out_dtype)

    def row(self, index: int, out_dtype=np.float32) -> np.ndarray:
        return self.gather(np.array([index]), out_dtype=out_dtype)[0]

    def __len__(self) -> int:
        return self.spec.n_rows

    def __repr__(self) -> str:
        mode = "resident" if self.resident else "memmap"
        return (
            f"GGUFNGramTable({os.path.basename(self.gguf_path)!r}, {self.tensor_name!r}, "
            f"rows={self.spec.n_rows}, {self.n_bytes / 1e9:.2f} GB, {mode})"
        )
