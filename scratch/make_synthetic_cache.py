"""Build a structurally-valid tiny teacher cache, so the training path can be
exercised before the real 1M capture finishes.

Not a substitute for real signals -- the values are noise. The point is that every
shape, dtype, manifest field and shard offset matches what OfflineTeacherCache
validates, so config/wiring errors surface now rather than after an 80-minute capture.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from distillkit.offline_cache import OfflineCacheWriter, file_sha256, tokenizer_vocab_hash

STUDENT = r"D:\DeepThought\Projects\HybridModel\student-hf"
OUT = r"D:\DeepThought\Projects\HybridModel\capture-synthetic"
HIDDEN, VOCAB, TOPK, ANCHORS = 5120, 248320, 64, [8, 64]

tok = AutoTokenizer.from_pretrained(STUDENT)
rng = np.random.default_rng(0)

with OfflineCacheWriter(
    OUT,
    tokenizer_hash=file_sha256(os.path.join(STUDENT, "tokenizer.json")),
    tokenizer_vocab_fingerprint=tokenizer_vocab_hash(tok),
    anchor_layers=ANCHORS, hidden_size=HIDDEN, vocab_size=VOCAB,
    sequence_length=1024, top_k=TOPK, shard_tokens=4096,
    metadata={"synthetic": True, "purpose": "training-path dry run"},
) as writer:
    for i in range(12):
        n = int(rng.integers(48, 160))
        ids = rng.integers(0, VOCAB, size=n).astype(np.int64)
        topk_ids = rng.integers(0, VOCAB, size=(n, TOPK)).astype("<u4")
        # Valid logprobs: descending and summing to < 1 in probability space.
        raw = np.sort(rng.standard_normal((n, TOPK)).astype(np.float32), axis=1)[:, ::-1]
        logprobs = (raw - np.log(np.exp(raw).sum(1, keepdims=True) * 1.4)).astype("<f2")
        # fp8 range is +-448; keep well inside so the writer's overflow guard passes.
        states = (rng.standard_normal((n, len(ANCHORS), HIDDEN)) * 2.0).astype(np.float32)
        import torch
        packed = torch.from_numpy(states).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
        writer.append(f"syn-{i:03d}", ids, topk_ids, logprobs, packed,
                      split="eval" if i % 4 == 0 else "train", original_length=n)
print("wrote", OUT)

from distillkit.offline_cache import OfflineTeacherCache
cache = OfflineTeacherCache(OUT)
print("validates OK:", len(cache.manifest["documents"]), "documents",
      {s: len(cache.document_ids(s)) for s in ("train", "eval")})
