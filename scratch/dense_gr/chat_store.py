"""Write the capture corpus into the token-store layout the conversion calibrates from.

`convert_full.py` fits every MLA and CSA2 projection by least squares against what the
source model's attention produced on calibration windows, and those windows come from
`scratch/code_training/tokens-v2` -- `bigcode/the-stack-v2-dedup`, config Python. The
conversion record says `calibration_tokens = 32768`: the whole refit of a 1.9B model was
solved from 32 windows of Python source, and then judged on MMLU's 57 academic subjects
and a chat corpus.

That is a domain mismatch in the conversion itself, and it is separable from the sample
size. This writes the same layout from `teacher-cache-5m`, whose documents are chat
throughout and carry the reasoning and multiple-choice shapes MMLU asks for, so the two
can be compared at the same token count before anything else changes.

The store is two flat uint32 files, which is what `open_split` memory-maps. Documents are
written end to end in the cache's own order; the conversion draws windows from the stream
rather than from documents, so a window can span a boundary. That is what the Python store
does too, and it is the comparison being made.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from distillkit.offline_cache import OfflineTeacherCache


def write_split(cache, split, destination, limit=None):
    ids = cache.document_ids(split)
    written = 0
    with open(destination, "wb") as handle:
        for doc_id in ids:
            tokens = np.asarray(
                cache.read_document(doc_id, tokens_only=True)["input_ids"],
                dtype=np.uint32)
            handle.write(tokens.tobytes())
            written += tokens.size
            if limit and written >= limit:
                break
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="../teacher-cache-5m")
    parser.add_argument("--output", type=Path,
                        default=Path("scratch/dense_gr/chat-tokens"))
    args = parser.parse_args()

    cache = OfflineTeacherCache(args.cache)
    args.output.mkdir(parents=True, exist_ok=True)
    # `train` is what the conversion calibrates from and `calibration` is what it scores
    # its own held-out loss on, which is the opposite of what those names suggest.
    train = write_split(cache, "train", args.output / "train.bin")
    held = write_split(cache, "eval", args.output / "calibration.bin")
    cache.close()
    print("wrote %s: train %d tokens, calibration %d tokens"
          % (args.output, train, held))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
