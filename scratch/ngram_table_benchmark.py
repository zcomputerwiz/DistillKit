"""Measure the actual IQ4_NL table at the planned 8 x 4096 token batch size.

This is an isolated provider benchmark, not a claim of overlap with training.
Use --resident to load 28.8 GB into process memory; otherwise prefault the memmap.
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import psutil
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distillkit.ngram_hash import NGramHasher
from distillkit.ngram_table import GGUFNGramTable, IQ4NLDequant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--resident", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.batch_size, args.sequence_length, args.repeats) <= 0:
        parser.error("batch size, sequence length and repeats must be positive")
    table = GGUFNGramTable(args.gguf)
    available = psutil.virtual_memory().available
    if available < table.n_bytes + (8 << 30):
        raise RuntimeError("Insufficient available RAM for the table plus 8 GiB benchmark headroom")
    print(f"Preparing {table.n_bytes / 1e9:.2f} GB table; available RAM {available / 1e9:.1f} GB", flush=True)
    prepare_seconds = table.load_resident() if args.resident else table.prefault()
    print(f"Table ready in {prepare_seconds:.2f}s; resident={table.resident}", flush=True)
    hasher = NGramHasher()
    generator = torch.Generator().manual_seed(1234)
    dequant = IQ4NLDequant().cuda() if torch.cuda.is_available() else None
    results = []
    for repeat in range(args.repeats):
        ids = torch.randint(0, hasher.config.vocab_size,
                            (args.batch_size, args.sequence_length), generator=generator)
        t0 = time.perf_counter()
        rows = hasher.row_indices(ids)
        hash_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        raw = table.gather_raw(rows)
        gather_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        host = table.gather(rows, out_dtype=np.float32)
        host_seconds = time.perf_counter() - t0
        assert np.isfinite(host).all()
        del host
        entry = {"hash_seconds": hash_seconds, "gather_seconds": gather_seconds,
                 "gather_cpu_dequant_seconds": host_seconds}
        if dequant is not None:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            raw = table.gather_raw(rows)
            pinned = torch.from_numpy(raw).pin_memory()
            out = dequant(pinned.to("cuda", non_blocking=True))
            torch.cuda.synchronize()
            entry["gather_pin_transfer_gpu_dequant_seconds"] = time.perf_counter() - t0
            assert tuple(out.shape) == (args.batch_size, args.sequence_length, 16, 160)
            assert torch.isfinite(out).all().item()
            del pinned, out
        results.append(entry)
        print(f"Iteration {repeat + 1}: {entry}", flush=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump({"gguf": args.gguf, "resident": table.resident,
                   "platform": platform.platform(), "torch": torch.__version__,
                   "batch_size": args.batch_size, "sequence_length": args.sequence_length,
                   "prepare_seconds": prepare_seconds, "measurements": results,
                   "training_overlap_verified": False}, fh, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    main()
