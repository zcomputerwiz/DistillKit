"""What does a given `ngram_vocab_size_base` actually do to this corpus?

The native table's address space is a choice, and a bad one is either mostly untouched
(paying for rows nothing trains) or heavily collided (rows that mean several different
n-grams at once). This runs the corpus through the hasher alone -- no model, no training,
no table -- and reports the occupancy, so the base can be chosen from a measurement rather
than from a round number.

It does not change the configured size. It reports and stops.

    python scratch/native_table/hash_survey.py --base 131072 --documents 2048
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from distillkit.ngram_hash import NGramHashConfig, NGramHasher

ROOT = Path(__file__).resolve().parents[2]
STUDENT = str((ROOT / ".." / "student-hf").resolve())
DOCUMENTS = str((ROOT / ".." / "capture-data" / "heldout.jsonl").resolve())


def survey(hasher, documents, tokenizer, limit):
    """Row touch counts over the corpus, per head and in total."""
    heads = hasher.config.ngram_heads
    counts = np.zeros(hasher.padded_vocab_size, dtype=np.int64)
    per_head = np.zeros(heads, dtype=np.int64)
    head_unique = [set() for _ in range(heads)]
    # Distinct n-gram identities, so a genuine collision -- two different n-grams sharing
    # a row -- can be told apart from the same n-gram recurring. Touches per used row
    # conflates the two.
    identities = {"bigram": set(), "trigram": set()}
    vocab = hasher.config.vocab_size
    touches = 0
    for text in documents:
        ids = tokenizer(text)["input_ids"][:limit]
        if len(ids) < 2:
            continue
        rows = hasher.row_indices(torch.tensor([ids], dtype=torch.long))[0].numpy()
        np.add.at(counts, rows.reshape(-1), 1)
        touches += rows.size
        for head in range(heads):
            per_head[head] += rows.shape[0]
            head_unique[head].update(rows[:, head].tolist())
        tokens = np.asarray(ids, dtype=np.int64)
        identities["bigram"].update((tokens[:-1] * vocab + tokens[1:]).tolist())
        if len(tokens) > 2:
            identities["trigram"].update(
                (tokens[:-2] * vocab * vocab + tokens[1:-1] * vocab + tokens[2:]).tolist())
    used = counts[counts > 0]
    bigram_heads = hasher.config.heads_per_ngram
    return {
        "base": hasher.config.ngram_vocab_size_base,
        "padded_vocab_size": hasher.padded_vocab_size,
        "total_vocab_size": hasher.total_vocab_size,
        "head_dim": hasher.config.head_dim,
        "table_parameters": hasher.padded_vocab_size * hasher.config.head_dim,
        "total_touches": int(touches),
        "unique_rows": int(used.size),
        "bigram_unique_rows": len(set().union(*head_unique[:bigram_heads])),
        "trigram_unique_rows": len(set().union(*head_unique[bigram_heads:])),
        "mean_touches_per_used_row": float(used.mean()) if used.size else 0.0,
        "median": float(np.median(used)) if used.size else 0.0,
        "p90": float(np.quantile(used, 0.90)) if used.size else 0.0,
        "p99": float(np.quantile(used, 0.99)) if used.size else 0.0,
        "max": int(used.max()) if used.size else 0,
        "singleton_fraction": float((used == 1).mean()) if used.size else 0.0,
        "table_fraction_touched": float(used.size / hasher.padded_vocab_size),
        "per_head_unique": [len(rows) for rows in head_unique],
        # Per head, because each head is its own address space: distinct n-grams seen
        # against distinct rows they landed on. 1 - rows/ngrams is the share of n-grams
        # that had to share a row with a different n-gram.
        "distinct_bigrams": len(identities["bigram"]),
        "distinct_trigrams": len(identities["trigram"]),
        "bigram_collision_rate": 1 - float(np.mean([len(head_unique[h]) for h in range(bigram_heads)])
                                           / max(len(identities["bigram"]), 1)),
        "trigram_collision_rate": 1 - float(np.mean([len(head_unique[h]) for h in range(bigram_heads, heads)])
                                            / max(len(identities["trigram"]), 1)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=int, nargs="+", default=[131072])
    parser.add_argument("--documents", type=int, default=2048)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(STUDENT, local_files_only=True)
    texts = []
    with open(DOCUMENTS, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                texts.append(json.loads(line)["text"])
            if len(texts) >= args.documents:
                break
    print("%d documents, capped at %d tokens each" % (len(texts), args.tokens))

    results = []
    for base in args.base:
        hasher = NGramHasher(NGramHashConfig(vocab_size=tokenizer.vocab_size,
                                             ngram_vocab_size_base=base,
                                             ple_embed_dim=args.embed_dim))
        stats = survey(hasher, texts, tokenizer, args.tokens)
        results.append(stats)
        print("\nbase %d: %d rows, %d parameters at %d dims"
              % (base, stats["padded_vocab_size"], stats["table_parameters"],
                 stats["head_dim"]))
        print("  touches %d over %d unique rows (%.2f%% of the table)"
              % (stats["total_touches"], stats["unique_rows"],
                 100 * stats["table_fraction_touched"]))
        print("  bigram unique %d, trigram unique %d"
              % (stats["bigram_unique_rows"], stats["trigram_unique_rows"]))
        print("  touches per used row: mean %.2f  median %.0f  p90 %.0f  p99 %.0f  max %d"
              % (stats["mean_touches_per_used_row"], stats["median"], stats["p90"],
                 stats["p99"], stats["max"]))
        print("  singletons %.1f%%" % (100 * stats["singleton_fraction"]))
        print("  distinct n-grams: %d bigram, %d trigram"
              % (stats["distinct_bigrams"], stats["distinct_trigrams"]))
        print("  collision rate per head: bigram %.1f%%  trigram %.1f%%"
              % (100 * stats["bigram_collision_rate"],
                 100 * stats["trigram_collision_rate"]))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
