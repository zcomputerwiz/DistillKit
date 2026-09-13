"""Build a per-context FFN residual cache from documents the evaluator never sees.

Two passes, because the memory arithmetic demands it: a full-vector prototype is 2048
values, so keeping one for every trigram in a six-million-token corpus would be tens of
gigabytes. The first pass counts contexts without the model and keeps only those that
recur across documents; the second runs the model and accumulates residuals for those.

Cross-document recurrence is the requirement, not raw frequency. A trigram appearing forty
times inside one document tells us nothing about whether a cache built elsewhere
transfers, and counting it would quietly measure within-document leakage.

Stored per key, per layer: the running sum, the sum of squared norms, and the count. The
mean is the prototype; the two together give the variance that the substitution thresholds
use as a confidence estimate.

    python scratch/ffn_memo/build_cache.py --documents 3000 --output scratch/ffn_memo/cache
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from distillkit.ffn_skip import capture_ffn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repeatability import (DEFAULT_BUNDLE, DEFAULT_MODEL, DEFAULT_SOURCE, LAYERS,
                           cache_documents, held_out_digests, trigram_keys)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--documents", type=int, default=3000)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--max-keys", type=int, default=400000)
    parser.add_argument("--layers", type=int, nargs="+", default=list(LAYERS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise SystemExit("mask to one card with CUDA_VISIBLE_DEVICES")

    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    excluded = held_out_digests(args.bundle)
    documents = cache_documents(args.source, tokenizer, excluded,
                                args.documents, args.length)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.use_cache = False
    vocab, hidden = config.vocab_size, config.hidden_size

    counts = collections.Counter()
    documents_per_key = collections.defaultdict(set)
    all_keys = []
    for number, ids in enumerate(documents):
        keys = trigram_keys(ids, vocab)
        all_keys.append(keys)
        for key in keys[2:]:
            counts[key] += 1
            documents_per_key[key].add(number)
    frequent = [key for key, count in counts.most_common() if count >= args.min_count]
    frequent = [key for key in frequent if len(documents_per_key[key]) >= 2][:args.max_keys]
    index_of = {key: position for position, key in enumerate(frequent)}
    tokens_total = sum(len(ids) for ids in documents)
    print("%d documents, %d tokens, %d distinct trigrams, %d cached keys"
          % (len(documents), tokens_total, len(counts), len(frequent)), flush=True)

    model = Qwen3_5ForCausalLM.from_pretrained(
        args.model, config=config, dtype=torch.bfloat16,
        local_files_only=True).to(args.device).eval()

    sums = {layer: torch.zeros(len(frequent), hidden, dtype=torch.float32)
            for layer in args.layers}
    square_sums = {layer: torch.zeros(len(frequent), dtype=torch.float64)
                   for layer in args.layers}
    global_sum = {layer: torch.zeros(hidden, dtype=torch.float64) for layer in args.layers}
    seen = torch.zeros(len(frequent), dtype=torch.int64)

    with torch.inference_mode():
        for number, (ids, keys) in enumerate(zip(documents, all_keys)):
            positions = [index for index in range(2, len(ids)) if keys[index] in index_of]
            tokens = torch.tensor([ids], device=args.device)
            with capture_ffn(model, args.layers) as captured:
                model(input_ids=tokens, attention_mask=torch.ones_like(tokens))
                for layer in args.layers:
                    values = captured[layer][0][1][0].float().cpu()
                    global_sum[layer] += values.to(torch.float64).sum(0)
                    if positions:
                        rows = torch.tensor([index_of[keys[index]] for index in positions])
                        picked = values[positions]
                        sums[layer].index_add_(0, rows, picked)
                        square_sums[layer].index_add_(
                            0, rows, picked.to(torch.float64).pow(2).sum(-1))
            if positions:
                seen.index_add_(0, torch.tensor([index_of[keys[i]] for i in positions]),
                                torch.ones(len(positions), dtype=torch.int64))
            if (number + 1) % 250 == 0:
                print("  %d/%d documents" % (number + 1, len(documents)), flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    counts_array = seen.numpy()
    keep = counts_array > 0
    kept_keys = np.array(frequent, dtype=np.int64)[keep]
    for layer in args.layers:
        mean = (sums[layer].numpy()[keep]
                / counts_array[keep][:, None].astype(np.float32))
        # Residual variance per key: E||r||^2 - ||mean||^2, which is what the confidence
        # threshold reads. Stored as a scalar; a full covariance would be 4 MB an entry.
        variance = (square_sums[layer].numpy()[keep] / counts_array[keep]
                    - (mean.astype(np.float64) ** 2).sum(-1))
        np.savez(args.output / ("layer-%d.npz" % layer),
                 keys=kept_keys, counts=counts_array[keep],
                 mean=mean.astype(np.float16), variance=variance,
                 global_mean=(global_sum[layer] / tokens_total).numpy().astype(np.float32))
        print("layer %d: %d entries, %.1f MB"
              % (layer, len(kept_keys), mean.astype(np.float16).nbytes / 2**20), flush=True)

    manifest = {"model": args.model, "source": args.source,
                "documents": len(documents), "tokens": tokens_total,
                "distinct_trigrams": len(counts), "cached_keys": int(keep.sum()),
                "min_count": args.min_count, "layers": args.layers,
                "hidden": hidden, "elapsed_seconds": time.monotonic() - started,
                "excluded_digests": len(excluded)}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                               encoding="utf-8")
    print("built in %.0f s" % manifest["elapsed_seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
