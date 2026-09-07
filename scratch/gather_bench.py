"""Can the §3a memmap row-gather keep up with training?

16 rows/token x batch 8 x 4096 = 524,288 row fetches of 160 B each, per step.
The spec's plan is: compute indices in the collator, gather on the host from a
memmap, pin, transfer. This measures whether that actually overlaps with compute.

Uses a 16 GB stand-in table rather than the real 51.2 GB one. That is representative
of the *warm* case the spec is banking on ("sits in page cache once warm") since both
fit in 128 GB RAM; it does NOT measure the cold-start NVMe cost.
"""
import os, sys, time
import numpy as np
import torch

TABLE_PATH = r"D:\DeepThought\Projects\HybridModel\DistillKit\scratch\ngram_table_bench.u8"
ROW_DIM = 160
N_ROWS = 100_000_000           # 16 GB at 160 B/row
BATCH, SEQ, HEADS = 8, 4096, 16
N_FETCH = BATCH * SEQ * HEADS

def make_table():
    size = N_ROWS * ROW_DIM
    if os.path.exists(TABLE_PATH) and os.path.getsize(TABLE_PATH) == size:
        print(f"reusing table ({size/1e9:.1f} GB)"); return
    print(f"creating {size/1e9:.1f} GB table at {TABLE_PATH} ...")
    t0 = time.time()
    mm = np.memmap(TABLE_PATH, dtype=np.uint8, mode="w+", shape=(N_ROWS, ROW_DIM))
    chunk = 1_000_000
    pattern = np.arange(chunk * ROW_DIM, dtype=np.uint64).astype(np.uint8).reshape(chunk, ROW_DIM)
    for start in range(0, N_ROWS, chunk):
        end = min(start + chunk, N_ROWS)
        mm[start:end] = pattern[: end - start]
    mm.flush(); del mm
    print(f"  wrote in {time.time()-t0:.1f}s")

def bench():
    table = np.memmap(TABLE_PATH, dtype=np.uint8, mode="r", shape=(N_ROWS, ROW_DIM))

    rng = np.random.default_rng(0)
    idx = rng.integers(0, N_ROWS, size=N_FETCH, dtype=np.int64)

    print("\nwarming page cache over the touched rows (2 passes)...")
    for _ in range(2):
        _ = table[idx]

    def timed(fn, n=5):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return min(ts), sum(ts)/len(ts)

    # 1. pure host gather
    best, avg = timed(lambda: table[idx])
    print(f"\n[gather]        best {best*1000:7.1f} ms  avg {avg*1000:7.1f} ms"
          f"  -> {N_FETCH/best/1e6:.1f} M rows/s, {N_FETCH*ROW_DIM/best/1e9:.2f} GB/s")
    gather_ms = best * 1000

    # 2. gather + pin + H2D + fp8->bf16
    if torch.cuda.is_available():
        stream = torch.cuda.Stream()
        def full():
            host = torch.from_numpy(np.ascontiguousarray(table[idx])).pin_memory()
            with torch.cuda.stream(stream):
                dev = host.to("cuda", non_blocking=True)
                out = dev.view(torch.float8_e4m3fn).to(torch.bfloat16)
            stream.synchronize()
            return out
        out = full()
        best_f, avg_f = timed(full)
        print(f"[gather+H2D+up] best {best_f*1000:7.1f} ms  avg {avg_f*1000:7.1f} ms"
              f"  -> out {tuple(out.shape)} {out.dtype}")
        print(f"                 non-gather overhead: {(best_f-best)*1000:.1f} ms")

    # 3. threaded gather
    from concurrent.futures import ThreadPoolExecutor
    for workers in (4, 8, 16):
        parts = np.array_split(idx, workers)
        with ThreadPoolExecutor(workers) as ex:
            def threaded():
                list(ex.map(lambda p: table[p], parts))
            best_t, _ = timed(threaded, n=3)
        print(f"[gather x{workers:2d}]     best {best_t*1000:7.1f} ms"
              f"  -> {N_FETCH/best_t/1e6:.1f} M rows/s  (speedup {gather_ms/(best_t*1000):.2f}x)")

    # 4. what is the compute budget we must hide under?
    print("\n--- budget ---")
    print(f"  fetches/step: {N_FETCH:,} rows x {ROW_DIM} B = {N_FETCH*ROW_DIM/1e6:.0f} MB gathered")
    for tps in (1000, 2000, 4000):
        step_s = BATCH*SEQ/tps
        print(f"  at {tps:5d} tok/s a step takes {step_s*1000:7.0f} ms"
              f"  -> gather is {gather_ms/(step_s*1000)*100:5.1f}% of it")

if __name__ == "__main__":
    make_table(); bench()
