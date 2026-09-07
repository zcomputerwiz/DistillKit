"""Sidecar lookup cost at speculative-decoding scale.

During generation the n-gram rows cannot be precomputed: the hash depends on tokens
the model just produced, so the gather sits on the critical path of every step.

With speculative decoding the target model verifies K draft tokens per forward, so a
step needs K x 16 rows instead of 16. The question is whether bigger K makes that
worse. Note total rows for N output tokens is 16N regardless of K -- speculation
changes how the work is batched, not how much there is, except that rows fetched for
*rejected* drafts are wasted, which scales as 1/acceptance_rate.

Measures cold (first touch, page-fault bound) and warm (pages resident) at K sizes a
real speculative decoder would use.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distillkit.ngram_table import GGUFNGramTable

GGUF = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "models--unsloth--Qwen3.8-Flash-Next-GGUF", "snapshots",
    "38bb39ee97821de2c9009abb7e93950eec396e66", "UD-IQ4_XS",
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf")

table = GGUFNGramTable(GGUF)
rng = np.random.default_rng(0)
HEADS = 16
N_ROWS = table.spec.n_real_rows

def timed(rows_idx, n=30):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        _ = table.raw[rows_idx]
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))

print(f"table: {table.n_bytes/1e9:.1f} GB memmap, {HEADS} rows/token\n")
print(f"{'K (draft tokens)':>17s} {'rows/step':>10s} {'COLD ms':>9s} {'WARM ms':>9s}")
results = {}
for k in (1, 4, 8, 16, 32):
    n = k * HEADS
    cold_idx = rng.integers(0, N_ROWS, size=n)          # never touched before
    t0 = time.perf_counter(); _ = table.raw[cold_idx]; cold = time.perf_counter() - t0
    warm = timed(cold_idx)                               # same rows, now resident
    results[k] = (cold, warm)
    print(f"{k:17d} {n:10d} {cold*1000:9.2f} {warm*1000:9.3f}")

print("\nper output token, assuming every draft token is accepted:")
for k, (cold, warm) in results.items():
    print(f"  K={k:2d}: cold {cold/k*1000:7.3f} ms/token   warm {warm/k*1000:7.4f} ms/token")

print("\nwasted-lookup penalty (rows are fetched for rejected drafts too):")
for acc in (0.5, 0.7, 0.9):
    k = 8
    warm = results[k][1]
    print(f"  K=8, {acc:.0%} acceptance -> {warm/(k*acc)*1000:.4f} ms per *accepted* token warm")

print("\nreference: a 4B decode step is roughly 20-40 ms on this class of GPU.")
