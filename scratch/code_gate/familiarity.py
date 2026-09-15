"""Build a Python trigram familiarity cache with the canonical definition, train split only.

The general-domain gate failed on Python for a concrete and measurable reason: 84.4% of
Python targets are trigrams the general-text cache never saw, so the learned
familiarity-conditioned policy has almost nothing to condition on and emits a near-constant
admission. This rebuilds the same statistic from Python instead, changing the source corpus
and nothing else about the definition.

Everything that makes the statistic what it is is copied from ``scratch/ffn_memo/build_cache.py``
rather than re-derived:

    key             (t-2, t-1, t) packed as (a * V + b) * V + c, the context whose FFN is
                    being computed -- exact, not hashed, so there are no collisions
    admission       a key is cached only if it recurs at least ``min_count`` times AND
                    appears in at least two documents. Cross-document recurrence is the
                    requirement: a trigram appearing forty times inside one file says
                    nothing about whether a cache transfers, and counting it would measure
                    within-document leakage
    statistics      per key, the mean FFN output vector and the residual variance
                    E||r||^2 - ||mean||^2, which is what the gate's second feature reads
                    after normalizing by the prototype's energy

Two passes, for the same reason the original has two: a full prototype is 2048 values, so
keeping one per trigram across 30M tokens would be tens of gigabytes. The first pass counts
without the model; the second runs the model only for the keys that survived.

**Only the train split is read.** Calibration and heldout never enter the cache, which is
the property that makes a later heldout number mean anything.

One deliberate deviation from the general cache, stated because it is a real choice: the
FFN prototypes are captured from **B_code**, the backbone this gate will actually be
attached to, not from B0. The variance feature is a claim about how consistent *this*
model's FFN output is for a context; computed from a different model it would describe
something the gate never sees. The count feature -- which is the one the 84.4% diagnostic is
about -- is a property of the corpus and is identical either way.

    python scratch/code_gate/familiarity.py --count-only
    CUDA_VISIBLE_DEVICES=0 python scratch/code_gate/familiarity.py
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code_training"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffn_memo"))

import numpy as np
import torch

from corpus import TOKENS, TokenStore, load_config

BCODE = Path("scratch/code_training/checkpoints/v1/final")
OUTPUT = Path("scratch/code_gate/cache")
LAYER = 12
#: Buckets from the original intervention study: the edges straddle 400 because that is
#: where the benefit turned positive there, not because they are evenly spaced.
COUNT_EDGES = (0, 1, 4, 100, 400, 800, 3000)


def trigram_keys(ids, vocab: int):
    """Key for position t is (t-2, t-1, t) -- the context whose FFN is being computed.

    Vectorized over a whole document; the reference implementation loops. Both produce
    int64 and the test asserts they agree exactly.
    """
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) < 3:
        return np.empty(0, dtype=np.int64)
    return (ids[:-2] * vocab + ids[1:-1]) * vocab + ids[2:]


def count_pass(store: TokenStore, vocab: int, min_count: int, max_keys: int,
               limit: int = 0, max_length: int = 0):
    """Trigram counts and cross-document occupancy over the train split, without the model.

    Sorted arrays rather than the reference's dict-of-sets. Over 30M tokens the reference
    shape would hold roughly ten million Python ints and ten million set objects -- several
    gigabytes, and slow enough to dominate the whole build. The arithmetic is identical and
    the test asserts the two agree on a corpus small enough to run both.
    """
    total = len(store) if limit <= 0 else min(limit, len(store))
    keys_per_document, owners, tokens = [], [], 0
    for number in range(total):
        ids = np.asarray(store.document(number), dtype=np.int64)
        # Truncated exactly as the capture pass truncates, so the counts a key is
        # admitted on are the counts its prototype is averaged over.
        if max_length:
            ids = ids[:max_length]
        tokens += len(ids)
        keys = trigram_keys(ids, vocab)
        if keys.size:
            keys_per_document.append(keys)
            owners.append(np.full(keys.size, number, dtype=np.int32))
    keys_all = np.concatenate(keys_per_document)
    owners_all = np.concatenate(owners)

    order = np.lexsort((owners_all, keys_all))
    keys_sorted = keys_all[order]
    owners_sorted = owners_all[order]
    unique, starts, occurrences = np.unique(keys_sorted, return_index=True,
                                            return_counts=True)
    # Distinct documents per key: a (key, document) pair is new where either changes.
    fresh = np.empty(keys_sorted.size, dtype=bool)
    fresh[0] = True
    fresh[1:] = (keys_sorted[1:] != keys_sorted[:-1]) | (owners_sorted[1:] != owners_sorted[:-1])
    document_counts = np.add.reduceat(fresh, starts)

    eligible = (occurrences >= min_count) & (document_counts >= 2)
    candidates = unique[eligible]
    candidate_counts = occurrences[eligible]
    # Most frequent first, matching the reference's ``most_common`` ordering before the cap.
    ranked = np.argsort(-candidate_counts, kind="stable")[:max_keys]
    frequent = candidates[ranked].tolist()
    counts = dict(zip(unique.tolist(), occurrences.tolist()))
    return counts, frequent, tokens, total


def bucket_occupancy(store: TokenStore, indices, vocab: int, lookup) -> dict:
    """Share of evaluation targets falling in each familiarity bucket under ``lookup``."""
    table = np.asarray(sorted(lookup), dtype=np.int64)
    values = np.asarray([lookup[int(key)] for key in table], dtype=np.int64)
    pieces = []
    for index in indices:
        keys = trigram_keys(np.asarray(store.document(index), dtype=np.int64), vocab)
        if keys.size:
            pieces.append(keys)
    keys_all = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
    found = np.zeros(keys_all.size, dtype=np.int64)
    if table.size:
        slot = np.searchsorted(table, keys_all)
        slot = np.clip(slot, 0, table.size - 1)
        hit = table[slot] == keys_all
        found[hit] = values[slot[hit]]

    buckets = collections.Counter()
    edges = list(zip(COUNT_EDGES, COUNT_EDGES[1:] + (float("inf"),)))
    for low, high in edges:
        chosen = int(((found >= low) & (found < high)).sum())
        if chosen:
            buckets["[%g, %g)" % (low, high)] = chosen
    total = keys_all.size
    return {name: {"tokens": value, "share": value / max(total, 1)}
            for name, value in sorted(buckets.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=Path, default=BCODE)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--max-keys", type=int, default=400_000)
    parser.add_argument("--documents", type=int, default=0, help="0 uses the whole split")
    parser.add_argument("--batch-tokens", type=int, default=16384)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count-only", action="store_true",
                        help="occupancy diagnostic without running the model")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    config = load_config()
    vocab = config.vocab_size
    store = TokenStore(TOKENS, "train")

    started = time.monotonic()
    print("counting trigrams over the train split (train only -- calibration and heldout "
          "never enter this cache)", flush=True)
    counts, frequent, tokens, documents = count_pass(
        store, vocab, args.min_count, args.max_keys, args.documents, args.max_length)
    print("%d documents, %d tokens, %d distinct trigrams, %d cacheable keys"
          % (documents, tokens, len(counts), len(frequent)), flush=True)

    # Section 7: how much of the unseen mass does a Python cache actually recover?
    from corpus import evaluation_subset

    heldout = TokenStore(TOKENS, "heldout")
    indices, subset = evaluation_subset(heldout, 1_250_000)
    general = np.load("scratch/ffn_memo/cache/layer-12.npz")
    general_lookup = dict(zip(general["keys"].tolist(), general["counts"].tolist()))
    python_lookup = {key: counts[key] for key in frequent}

    occupancy = {
        "general": bucket_occupancy(heldout, indices, vocab, general_lookup),
        "python": bucket_occupancy(heldout, indices, vocab, python_lookup),
    }
    print("\n%-16s %14s %14s" % ("bucket", "general cache", "python cache"))
    for name in sorted(set(occupancy["general"]) | set(occupancy["python"])):
        a = occupancy["general"].get(name, {}).get("share", 0.0)
        b = occupancy["python"].get(name, {}).get("share", 0.0)
        print("%-16s %13.1f%% %13.1f%%" % (name, 100 * a, 100 * b))

    report = {
        "source": "code_corpus/v1 train split", "split": "train",
        "documents": documents, "tokens": tokens,
        "distinct_trigrams": len(counts), "cacheable_keys": len(frequent),
        "min_count": args.min_count, "max_keys": args.max_keys,
        "layer": LAYER, "backbone": str(args.backbone),
        "evaluation_subset": subset, "bucket_occupancy": occupancy,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "occupancy.json").write_text(json.dumps(report, indent=2),
                                                encoding="utf-8")
    if args.count_only:
        print("\nwrote %s (count-only; no cache built)" % (args.output / "occupancy.json"))
        return 0

    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from distillkit.ffn_skip import capture_ffn

    model_config = AutoConfig.from_pretrained(args.backbone, local_files_only=True)
    model_config = getattr(model_config, "text_config", model_config)
    model_config.use_cache = False
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.backbone, config=model_config, dtype=torch.bfloat16,
        local_files_only=True).to(args.device).eval()
    model.requires_grad_(False)

    index_of = {key: position for position, key in enumerate(frequent)}
    hidden = model_config.hidden_size
    sums = torch.zeros(len(frequent), hidden, dtype=torch.float32)
    square_sums = torch.zeros(len(frequent), dtype=torch.float64)
    global_sum = torch.zeros(hidden, dtype=torch.float64)
    seen = torch.zeros(len(frequent), dtype=torch.int64)
    captured_tokens = 0

    order = sorted(range(documents), key=lambda i: -(store.offsets[i + 1] - store.offsets[i]))
    with torch.inference_mode():
        position = 0
        while position < len(order):
            group, widest = [], 0
            while position < len(order):
                length = min(int(store.offsets[order[position] + 1]
                                 - store.offsets[order[position]]), args.max_length)
                candidate = max(widest, length)
                if group and candidate * (len(group) + 1) > args.batch_tokens:
                    break
                group.append(order[position])
                widest = candidate
                position += 1
            rows = [np.asarray(store.document(i)[:args.max_length], dtype=np.int64)
                    for i in group]
            width = max(len(r) for r in rows)
            ids = torch.full((len(rows), width), model_config.eos_token_id,
                             dtype=torch.long, device=args.device)
            mask = torch.zeros((len(rows), width), dtype=torch.long, device=args.device)
            for slot, row in enumerate(rows):
                ids[slot, :len(row)] = torch.from_numpy(row).to(args.device)
                mask[slot, :len(row)] = 1
            with capture_ffn(model, [LAYER]) as grabbed:
                model(input_ids=ids, attention_mask=mask)
                values = grabbed[LAYER][0][1].float().cpu()
            for slot, row in enumerate(rows):
                keys = trigram_keys(row, vocab)
                picked, target = [], []
                for offset, key in enumerate(keys):
                    slot_index = index_of.get(int(key))
                    if slot_index is not None:
                        picked.append(offset + 2)      # key at offset addresses token t
                        target.append(slot_index)
                global_sum += values[slot, :len(row)].to(torch.float64).sum(0)
                captured_tokens += len(row)
                if picked:
                    chosen = values[slot, picked]
                    rows_index = torch.tensor(target)
                    sums.index_add_(0, rows_index, chosen)
                    square_sums.index_add_(0, rows_index,
                                           chosen.to(torch.float64).pow(2).sum(-1))
                    seen.index_add_(0, rows_index,
                                    torch.ones(len(picked), dtype=torch.int64))
            if position % 2000 < len(group):
                print("  captured %d/%d documents (%.0f s)"
                      % (position, len(order), time.monotonic() - started), flush=True)

    counts_array = seen.numpy()
    keep = counts_array > 0
    kept_keys = np.array(frequent, dtype=np.int64)[keep]
    mean = sums.numpy()[keep] / counts_array[keep][:, None].astype(np.float32)
    variance = (square_sums.numpy()[keep] / counts_array[keep]
                - (mean.astype(np.float64) ** 2).sum(-1))
    np.savez(args.output / ("layer-%d.npz" % LAYER),
             keys=kept_keys, counts=counts_array[keep],
             mean=mean.astype(np.float16), variance=variance,
             global_mean=(global_sum / max(captured_tokens, 1)).numpy().astype(np.float32))

    report.update({"cached_keys": int(keep.sum()), "captured_tokens": captured_tokens,
                   "elapsed_seconds": time.monotonic() - started})
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2),
                                               encoding="utf-8")
    print("\nlayer %d: %d entries, %.1f MB, %.0f s"
          % (LAYER, len(kept_keys), mean.astype(np.float16).nbytes / 2 ** 20,
             report["elapsed_seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
