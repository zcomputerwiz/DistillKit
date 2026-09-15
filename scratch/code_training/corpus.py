"""Shared plumbing for the code-training experiments: identity, token store, classes.

One module so that the baseline evaluation, the training run and the post-training
evaluation cannot disagree about what the corpus is, which tokenizer produced it, or how
a target token is classified. Each of those is a place where two scripts drifting apart
would produce a comparison that looks valid and is not.

The token store is a flat ``uint32`` memmap per split plus a document-offset index. It is
written once and read by everything: the packed training stream is a slice of it, the
evaluation is a deterministic subset of it, and the maximum-token-id check that gates the
whole task is one pass over it. Re-tokenizing per consumer would be both slower and a way
for the training tokens and the evaluated tokens to stop being the same tokens.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from distillkit.code_classes import (CODE_CLASSES, HISTORICAL, HISTORICAL_CLASSES,
                                     build_code_classes)

BASE = "D:/DeepThought/Projects/HybridModel/student-2b-hf"
CORPUS = Path("scratch/code_corpus/v1")
TOKENS = Path("scratch/code_training/tokens")
SPLITS = ("train", "calibration", "heldout")


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(BASE, local_files_only=True)


def load_config():
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(BASE, local_files_only=True)
    return getattr(config, "text_config", config)


def corpus_identity() -> dict:
    """The frozen definition, stamped into every artifact this experiment produces.

    Section 3 of the task: dataset revision, manifest, seed, filters and tokenizer are
    fixed from here on, and an artifact that does not carry this identity cannot be
    compared with one that does.
    """
    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    return {
        "corpus": "code_corpus/v1",
        "dataset": manifest["dataset"], "config": manifest["config"],
        "dataset_revision": manifest["revision"],
        "split_seed": manifest["seed"],
        "split_algorithm": manifest["split_algorithm"],
        "filters": manifest["filters"],
        "tokenizer": manifest["tokenizer"],
        "manifest_sha256": hashlib.sha256(
            (CORPUS / "manifest.json").read_bytes()).hexdigest(),
        "splits": {name: manifest["splits"][name] for name in SPLITS},
    }


class TokenStore:
    """A split's tokens as one flat array, with document boundaries kept alongside.

    ``offsets[i]:offsets[i + 1]`` is document *i*, EOS included as its final token. The
    packed training stream reads straight through ``tokens``; the evaluation reads
    documents. Both therefore see byte-identical ids, which is the point of storing them
    rather than re-deriving them.
    """

    def __init__(self, root: Path, split: str):
        self.split = split
        self.tokens = np.memmap(root / ("%s.bin" % split), dtype=np.uint32, mode="r")
        self.offsets = np.load(root / ("%s.idx.npy" % split))
        self.meta = json.loads((root / ("%s.json" % split)).read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def document(self, index: int) -> np.ndarray:
        return self.tokens[self.offsets[index]:self.offsets[index + 1]]

    @property
    def total_tokens(self) -> int:
        return int(self.offsets[-1])


def evaluation_subset(store: TokenStore, target_tokens: int, seed: int = 20260914):
    """A deterministic document subset of a split, reused at every checkpoint.

    Selected by hashing each document's ``(repo_id, path)`` -- the same identity the
    corpus sample ranks by -- so the subset depends on nothing but the corpus and the
    seed. Not on iteration order, not on how many tokens were wanted at the time, and
    not on which checkpoint asked for it. Documents are then taken in that hash order
    until the token target is met, and the *resulting* order is restored to document
    index so scoring is stable.

    ``target_tokens <= 0`` means the whole split.
    """
    identities = store.meta["documents"]
    ranked = sorted(range(len(identities)),
                    key=lambda i: hashlib.blake2b(
                        ("%d:%s" % (seed, identities[i])).encode("utf-8"),
                        digest_size=8).digest())
    if target_tokens <= 0:
        chosen = list(range(len(identities)))
    else:
        chosen, total = [], 0
        for index in ranked:
            chosen.append(index)
            total += int(store.offsets[index + 1] - store.offsets[index])
            if total >= target_tokens:
                break
    chosen.sort()
    digest = hashlib.sha256(
        "\n".join(identities[i] for i in chosen).encode("utf-8")).hexdigest()
    tokens = sum(int(store.offsets[i + 1] - store.offsets[i]) for i in chosen)
    return chosen, {"documents": len(chosen), "tokens": tokens, "seed": seed,
                    "digest": digest, "split": store.split}


def class_tables(tokenizer, vocab_size: int):
    """Per-id code class and the historical class it rolls up into, as int arrays."""
    names = build_code_classes(tokenizer, vocab_size)
    code_index = {name: i for i, name in enumerate(CODE_CLASSES)}
    hist_index = {name: i for i, name in enumerate(HISTORICAL_CLASSES)}
    code = np.fromiter((code_index[n] for n in names), dtype=np.int16, count=vocab_size)
    historical = np.fromiter((hist_index[HISTORICAL[n]] for n in names),
                             dtype=np.int16, count=vocab_size)
    return code, historical
