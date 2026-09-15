"""Section 0 bookkeeping: prove the corpus tokens are the model's tokens.

Three separate things can be true or false here and each fails silently:

*The vocabulary numbers must reconcile.* ``tokenizer.vocab_size`` counts the base BPE
merges and excludes added tokens; ``len(tokenizer)`` includes them; the model's embedding
is padded above both to a hardware-friendly multiple. Those are three different numbers
that are all correct, and the only thing that matters is whether any id the corpus
actually contains indexes past the embedding.

*The corpus pipeline and the training path must agree bit for bit.* They are the same
call in this experiment -- the token store is what training reads -- but that has to be
demonstrated rather than asserted, because the repository's other entry points tokenize
with ``add_special_tokens=True`` and a BOS prepended to a packed stream would shift every
target by one.

*The stored ids must round-trip to the stored source.* A decode mismatch means the tokens
being trained on are not the text that was filtered, decontaminated and split.

    python scratch/code_training/identity_check.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pyarrow.parquet as pq

from corpus import CORPUS, SPLITS, TOKENS, TokenStore, load_config, load_tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=200, help="files per split")
    parser.add_argument("--output", type=Path,
                        default=Path("scratch/code_training/manifests/identity.json"))
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    config = load_config()
    rows = config.vocab_size

    report = {
        "tokenizer.vocab_size": tokenizer.vocab_size,
        "len(tokenizer)": len(tokenizer),
        "added_tokens": len(tokenizer.get_added_vocab()),
        "config.vocab_size": rows,
        "input_embedding_rows": None, "lm_head_rows": None,
        "tie_word_embeddings": bool(getattr(config, "tie_word_embeddings", False)),
        "splits": {}, "round_trip_failures": [], "path_mismatches": [],
    }

    from safetensors import safe_open

    for path in sorted(Path(CORPUS.parts[0] if False else
                            "D:/DeepThought/Projects/HybridModel/student-2b-hf").glob("*.safetensors")):
        with safe_open(path, "pt") as handle:
            for key in handle.keys():
                shape = list(handle.get_slice(key).get_shape())
                if key.endswith("embed_tokens.weight"):
                    report["input_embedding_rows"] = shape[0]
                if key.endswith("lm_head.weight"):
                    report["lm_head_rows"] = shape[0]
    if report["lm_head_rows"] is None and report["tie_word_embeddings"]:
        # Tied weights: there is no separate lm_head tensor, the embedding is transposed.
        report["lm_head_rows"] = report["input_embedding_rows"]

    print("tokenizer.vocab_size   %d   (base BPE, excludes added tokens)"
          % report["tokenizer.vocab_size"])
    print("len(tokenizer)         %d   (+%d added/special)"
          % (report["len(tokenizer)"], report["added_tokens"]))
    print("config.vocab_size      %d" % rows)
    print("input embedding rows   %s" % report["input_embedding_rows"])
    print("lm_head rows           %s%s" % (report["lm_head_rows"],
                                           " (tied)" if report["tie_word_embeddings"] else ""))

    for split in SPLITS:
        store = TokenStore(TOKENS, split)
        maximum = int(np.asarray(store.tokens).max())
        report["splits"][split] = {
            "documents": len(store), "tokens": store.total_tokens,
            "max_token_id": maximum, "within_embedding": maximum < rows,
        }
        print("%-12s max token id %d  < %d: %s"
              % (split, maximum, rows, maximum < rows))

        # Bit-identity: re-read the source from parquet, tokenize it the way the corpus
        # builder did, and compare against what the training store actually holds.
        seen = 0
        for part in sorted((CORPUS / split).glob("part-*.parquet")):
            table = pq.read_table(part, columns=["repo_id", "path", "source"]).to_pylist()
            for offset, row in enumerate(table):
                if seen >= args.sample:
                    break
                fresh = tokenizer(row["source"], add_special_tokens=False)["input_ids"]
                stored = store.document(seen)
                # the store appends EOS as the document boundary; the rest must match
                if list(stored[:-1]) != fresh or int(stored[-1]) != tokenizer.eos_token_id:
                    report["path_mismatches"].append(row["repo_id"] + row["path"])
                if tokenizer.decode(fresh) != row["source"]:
                    report["round_trip_failures"].append(row["repo_id"] + row["path"])
                seen += 1
            if seen >= args.sample:
                break
        report["splits"][split]["sampled"] = seen

    within = all(entry["within_embedding"] for entry in report["splits"].values())
    clean = not report["path_mismatches"] and not report["round_trip_failures"]
    print("\nsampled %d files/split: %d pipeline mismatches, %d round-trip failures"
          % (args.sample, len(report["path_mismatches"]), len(report["round_trip_failures"])))
    report["verdict"] = ("vocab_size vs len(tokenizer) distinction plus embedding padding; "
                         "no rebuild required" if within and clean else "INCONSISTENT")
    print("verdict: %s" % report["verdict"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if within and clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
