"""Is this corpus in-distribution for our teacher, judged from the capture itself?

The capture stores the teacher's top-64 logprobs per position. Two things fall out
without reloading the 27B:

  * how often the actual next token is inside the teacher's top-64 -- if the corpus
    were badly off-distribution (wrong chat template, alien formatting) the teacher
    would be surprised constantly and coverage would collapse;
  * the teacher's logprob for the true next token when present, i.e. its confidence.

A well-matched corpus should sit high on both. This is the empirical form of "does
the dataset work".
"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distillkit.offline_cache import OfflineTeacherCache

cache = OfflineTeacherCache(r"D:\DeepThought\Projects\HybridModel\capture-smoke")
m = cache.manifest
print("manifest:")
for k in ("anchor_layers", "hidden_size", "vocab_size", "sequence_length", "top_k",
          "hidden_dtype", "hidden_layout"):
    print(f"  {k}: {m.get(k)}")
docs = m["documents"]
print(f"  documents: {len(docs)}  splits: {dict((s, sum(1 for d in docs if d['split']==s)) for s in {d['split'] for d in docs})}")
total_tokens = sum(d["length"] for d in docs)
print(f"  tokens: {total_tokens:,}")

hits = ranks = 0
lp_true, top1_lp = [], []
for doc in docs:
    rec = cache.read_document(doc["doc_id"])
    ids, tk_ids, tk_lp = rec["input_ids"], rec["topk_ids"], rec["topk_logprobs"]
    # position t predicts token t+1
    nxt = ids[1:]
    pred_ids, pred_lp = tk_ids[:-1], tk_lp[:-1].astype(np.float32)
    match = pred_ids == nxt[:, None]
    present = match.any(axis=1)
    hits += int(present.sum()); ranks += len(nxt)
    if present.any():
        idx = match[present].argmax(axis=1)
        lp_true.extend(pred_lp[present][np.arange(present.sum()), idx].tolist())
    top1_lp.extend(pred_lp[:, 0].tolist())

lp_true = np.array(lp_true); top1_lp = np.array(top1_lp)
print(f"\nnext token inside teacher top-64: {hits}/{ranks} = {100*hits/ranks:.2f}%")
print(f"mean logprob of true next token (when present): {lp_true.mean():.4f}"
      f"  -> perplexity {np.exp(-lp_true.mean()):.2f}")
print(f"mean top-1 logprob (teacher confidence):        {top1_lp.mean():.4f}")
print(f"median top-1 prob: {np.exp(np.median(top1_lp)):.3f}")
