"""How much of the Python corpus a truncated vocabulary would cover."""
import numpy as np

tokens = np.memmap("scratch/code_training/tokens/train.bin", dtype=np.uint32, mode="r")
total = tokens.shape[0]
counts = np.bincount(np.asarray(tokens, dtype=np.int64), minlength=248_320)
order = np.argsort(counts)[::-1]
ranked = counts[order]
cumulative = np.cumsum(ranked) / total

print("corpus tokens: %d" % total)
print("distinct ids used: %d of 248,320 (%.1f%%)"
      % (int((counts > 0).sum()), 100 * (counts > 0).sum() / 248_320))
print()
print("%10s %12s %12s" % ("top-N", "coverage", "miss rate"))
for n in (4096, 8192, 16_384, 32_768, 65_536, 131_072):
    if n <= ranked.shape[0]:
        cover = cumulative[n - 1]
        print("%10d %11.5f%% %11.5f%%" % (n, 100 * cover, 100 * (1 - cover)))
for target in (0.999, 0.9999, 0.99999, 1.0):
    need = int(np.searchsorted(cumulative, target) + 1)
    print("coverage %.5f needs top %d ids" % (target, need))
