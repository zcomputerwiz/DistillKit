"""Tokenize the corpus once into a flat store, and settle the vocabulary question.

Section 0 of the task is a gate: the corpus reported ``tokenizer.vocab_size = 248,044``
while the model's embedding is 248,320 rows, and until that is explained no training
should start. It is explained here by measurement rather than by argument -- the three
numbers are printed side by side, and the maximum token id actually present in each split
is compared against the embedding row count. An id at or above that count would index
past the embedding matrix, which is the failure this gate exists to catch.

The same pass writes the token store everything downstream reads. Tokenizing once and
storing the ids means the training stream and the evaluation stream are provably the same
tokens; re-tokenizing per consumer leaves room for them to quietly stop being so.

Each document ends with EOS, which is both the document boundary for packing and what
keeps a packed sequence from presenting the tail of one file as the context for the head
of an unrelated one.

    python scratch/code_training/tokenize_corpus.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pyarrow.parquet as pq

from corpus import BASE, CORPUS, SPLITS, TOKENS, corpus_identity, load_config, load_tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--output", type=Path, default=TOKENS)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    config = load_config()
    embedding_rows = config.vocab_size
    eos = tokenizer.eos_token_id

    vocabulary = {
        "tokenizer.vocab_size": tokenizer.vocab_size,
        "len(tokenizer)": len(tokenizer),
        "added_tokens": len(tokenizer.get_added_vocab()),
        "config.vocab_size": embedding_rows,
        "eos_token_id": eos,
    }
    print("vocabulary:")
    for key, value in vocabulary.items():
        print("  %-24s %d" % (key, value))

    args.output.mkdir(parents=True, exist_ok=True)
    report = {"vocabulary": vocabulary, "corpus": corpus_identity(), "splits": {}}
    started = time.monotonic()

    for split in SPLITS:
        ids_out = []
        offsets = [0]
        identities = []
        maximum = 0
        parts = sorted((args.corpus / split).glob("part-*.parquet"))
        for part in parts:
            table = pq.read_table(part, columns=["repo_id", "path", "source",
                                                 "token_count"]).to_pylist()
            for start in range(0, len(table), args.batch):
                chunk = table[start:start + args.batch]
                encoded = tokenizer([row["source"] for row in chunk],
                                    add_special_tokens=False)["input_ids"]
                for row, tokens in zip(chunk, encoded):
                    if len(tokens) != row["token_count"]:
                        raise SystemExit(
                            "token_count drift in %s%s: stored %d, retokenized %d"
                            % (row["repo_id"], row["path"], row["token_count"], len(tokens)))
                    tokens = tokens + [eos]
                    maximum = max(maximum, max(tokens))
                    ids_out.append(np.asarray(tokens, dtype=np.uint32))
                    offsets.append(offsets[-1] + len(tokens))
                    identities.append(row["repo_id"] + row["path"])

        flat = np.concatenate(ids_out)
        flat.tofile(args.output / ("%s.bin" % split))
        np.save(args.output / ("%s.idx.npy" % split), np.asarray(offsets, dtype=np.int64))
        entry = {
            "documents": identities,
            "tokens_with_eos": int(flat.size),
            "tokens_without_eos": int(flat.size - len(identities)),
            "max_token_id": int(maximum),
            "embedding_rows": embedding_rows,
            "within_embedding": bool(maximum < embedding_rows),
        }
        (args.output / ("%s.json" % split)).write_text(json.dumps(entry), encoding="utf-8")
        summary = {k: v for k, v in entry.items() if k != "documents"}
        summary["files"] = len(identities)
        report["splits"][split] = summary
        print("%-12s %7d files  %10d tokens (+%d EOS)  max id %d  < %d rows: %s"
              % (split, len(identities), entry["tokens_without_eos"], len(identities),
                 maximum, embedding_rows, entry["within_embedding"]))

    report["elapsed_seconds"] = time.monotonic() - started
    (args.output / "vocabulary.json").write_text(json.dumps(report, indent=2),
                                                 encoding="utf-8")

    ok = all(entry["within_embedding"] for entry in report["splits"].values())
    print("\nmax token id across all splits: %d, embedding rows: %d"
          % (max(e["max_token_id"] for e in report["splits"].values()), embedding_rows))
    print("GATE: %s" % ("PASS" if ok else "FAIL -- token ids exceed the embedding"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
