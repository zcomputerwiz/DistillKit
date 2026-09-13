"""Is the same FFN computation worth caching -- does a familiar context repeat its residual?

Memoisation only makes sense if the thing being memoised is stable. Before any cache,
basis, threshold grid or router, measure the quantity the whole idea rests on: across
separate occurrences of the same local context, how much of the FFN residual does a stored
mean actually explain?

Two numbers decide it, per layer:

    within-key   E|| r - mean(key) ||^2  /  E|| r ||^2
    global       E|| r - mean(all)  ||^2  /  E|| r ||^2

The second is what a single layer-global residual would leave unexplained -- the null
model that needs no cache at all. If the two are close, the key carries no information and
a cache keyed on local context is storing noise at 8 KB an entry.

The key is an exact token trigram, packed into one integer, so there are no hash
collisions to explain away a negative result. Occurrences are counted across documents:
a key seen many times inside one document proves nothing about generalisation, so the
accumulation also tracks how many distinct documents each key appeared in.

    python scratch/ffn_memo/repeatability.py --documents 1500 --output ...
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from distillkit.ffn_skip import capture_ffn

DEFAULT_MODEL = "D:/DeepThought/Projects/HybridModel/student-2b-hf"
DEFAULT_SOURCE = "D:/DeepThought/Projects/HybridModel/capture-data/heldout.jsonl"
DEFAULT_BUNDLE = "scratch/independent-eval/full-bundle-384.json"
LAYERS = (8, 12, 16, 20)


def held_out_digests(bundle_path):
    """Every document the evaluator will score, so the cache corpus can exclude them."""
    with io.open(bundle_path, encoding="utf-8") as handle:
        bundle = json.load(handle)
    digests = set()
    for split in bundle["splits"].values():
        for record in split.get("nll", []):
            if "text_sha256" in record:
                digests.add(record["text_sha256"])
    return digests


def cache_documents(source, tokenizer, excluded, count, length):
    """Tokenised windows from documents that are not in the evaluation bundle."""
    documents = []
    with io.open(source, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in excluded:
                continue
            ids = tokenizer(text)["input_ids"][:length]
            if len(ids) >= 8:
                documents.append(ids)
            if len(documents) >= count:
                break
    return documents


def trigram_keys(ids, vocab):
    """Key for position t is (t-2, t-1, t): the context whose FFN is being computed."""
    keys = [None, None]
    for index in range(2, len(ids)):
        keys.append((ids[index - 2] * vocab + ids[index - 1]) * vocab + ids[index])
    return keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--documents", type=int, default=1500)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--min-count", type=int, default=4)
    parser.add_argument("--max-keys", type=int, default=40000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
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
    vocab = config.vocab_size

    # Pass one: which contexts recur, and in how many different documents.
    counts = collections.Counter()
    documents_per_key = collections.defaultdict(set)
    all_keys = []
    for number, ids in enumerate(documents):
        keys = trigram_keys(ids, vocab)
        all_keys.append(keys)
        for key in keys[2:]:
            counts[key] += 1
            documents_per_key[key].add(number)
    frequent = [key for key, count in counts.most_common()
                if count >= args.min_count][:args.max_keys]
    # Cross-document only: a key that recurs inside one document says nothing about
    # whether a cache built elsewhere would transfer.
    frequent = [key for key in frequent if len(documents_per_key[key]) >= 2]
    index_of = {key: position for position, key in enumerate(frequent)}
    print("%d documents, %d distinct trigrams, %d usable keys (count>=%d, >=2 documents)"
          % (len(documents), len(counts), len(frequent), args.min_count), flush=True)
    if not frequent:
        raise SystemExit("no key recurs across documents; nothing to memoise")

    model = Qwen3_5ForCausalLM.from_pretrained(
        args.model, config=config, dtype=torch.bfloat16,
        local_files_only=True).to(args.device).eval()

    hidden = config.hidden_size
    sums = {layer: torch.zeros(len(frequent), hidden, dtype=torch.float64)
            for layer in LAYERS}
    square_sums = {layer: torch.zeros(len(frequent), dtype=torch.float64)
                   for layer in LAYERS}
    seen = torch.zeros(len(frequent), dtype=torch.int64)
    global_sum = {layer: torch.zeros(hidden, dtype=torch.float64) for layer in LAYERS}
    global_square = {layer: 0.0 for layer in LAYERS}
    global_count = 0

    with torch.inference_mode():
        for number, (ids, keys) in enumerate(zip(documents, all_keys)):
            positions = [index for index in range(2, len(ids)) if keys[index] in index_of]
            tokens = torch.tensor([ids], device=args.device)
            with capture_ffn(model, LAYERS) as captured:
                model(input_ids=tokens, attention_mask=torch.ones_like(tokens))
                for layer in LAYERS:
                    _, residual = captured[layer][0]
                    values = residual[0].to(torch.float64).cpu()
                    global_sum[layer] += values.sum(0)
                    global_square[layer] += float(values.pow(2).sum())
                    if positions:
                        rows = torch.tensor([index_of[keys[index]] for index in positions])
                        picked = values[positions]
                        sums[layer].index_add_(0, rows, picked)
                        square_sums[layer].index_add_(0, rows, picked.pow(2).sum(-1))
            global_count += len(ids)
            if positions:
                seen.index_add_(0, torch.tensor([index_of[keys[i]] for i in positions]),
                                torch.ones(len(positions), dtype=torch.int64))
            if (number + 1) % 100 == 0:
                print("  captured %d/%d documents" % (number + 1, len(documents)),
                      flush=True)

    report = {"model": args.model, "documents": len(documents),
              "distinct_trigrams": len(counts), "usable_keys": len(frequent),
              "min_count": args.min_count, "layers": list(LAYERS), "by_layer": {}}
    kept = seen >= 2
    print("\n%-6s %12s %12s %12s %10s" % ("layer", "within-key", "global-mean",
                                          "explained", "occurrences"))
    for layer in LAYERS:
        occurrences = seen[kept].to(torch.float64)
        means = sums[layer][kept] / occurrences.unsqueeze(-1)
        # Sum over occurrences of ||r||^2, and of ||mean||^2 counted once per occurrence:
        # E||r - mean||^2 = E||r||^2 - ||mean||^2 within each key.
        total_square = float(square_sums[layer][kept].sum())
        explained_square = float((means.pow(2).sum(-1) * occurrences).sum())
        within = (total_square - explained_square) / total_square

        global_mean = global_sum[layer] / global_count
        global_residual = (total_square
                           - 2 * float((sums[layer][kept] * global_mean).sum())
                           + float(global_mean.pow(2).sum()) * float(occurrences.sum()))
        global_relative = global_residual / total_square

        report["by_layer"][str(layer)] = {
            "within_key_relative_variance": within,
            "global_mean_relative_variance": global_relative,
            "explained_over_global": 1 - within / global_relative if global_relative else 0.0,
            "keys": int(kept.sum()), "occurrences": int(occurrences.sum()),
            "mean_residual_norm": (explained_square / float(occurrences.sum())) ** 0.5,
        }
        print("%-6d %12.4f %12.4f %12.4f %10d"
              % (layer, within, global_relative,
                 report["by_layer"][str(layer)]["explained_over_global"],
                 int(occurrences.sum())))

    report["elapsed_seconds"] = time.monotonic() - started
    print("\n%.0f s" % report["elapsed_seconds"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
