"""Build a general-text capture corpus, screened against every benchmark we score.

The chat corpus repaired what it contains: MMLU recovered to the source, and in-domain
NLL beat it by 0.58 nats, while NLL on general prose stayed 0.24 nats behind -- the
attention the conversion replaced was trained on broad text, and nothing in a chat-SFT
corpus retrains it. This builds the broad text: educational web prose, mathematical prose
and raw source files, rendered as plain text with no chat template, for the 27B teacher
to capture.

Documents are cut into pieces of at most `--piece` tokens rather than handed to the
capture whole. Training reads a 1024-token prefix, so a 4096-token document costs four
times the teacher compute and teaches a quarter of it; as pieces, every captured token is
one the trainer reads. A piece that starts mid-document is an ordinary pretraining
window, and the teacher sees exactly the same piece.

Every piece is screened, as `expand_corpus.py` screens, against the scored MMLU and ARC
test questions and -- because FineWeb contains Wikipedia copies and WikiText-103 test is
now the general-text yardstick -- against 13-word runs of WikiText-103 test itself.

    python scratch/dense_gr/general_corpus.py --output ../capture-data/general-pilot.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from expand_corpus import build_banks, fingerprint, shingles  # noqa: E402

TEACHER = "D:/DeepThought/Projects/HybridModel/teacher-hf"
CODE_STORE = Path(__file__).resolve().parent.parent / "code_training" / "tokens-v2"

SOURCES = [
    # label, repo, config, text field, token budget, domain
    ("fineweb-edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "text", 1_800_000, "prose"),
    ("finemath", "HuggingFaceTB/finemath", "finemath-4plus", "text", 600_000, "math"),
]


def wikitext_bank():
    from datasets import load_dataset

    text = " ".join(load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")["text"])
    bank = shingles(text, (13,))
    print("wikitext-103 test: %d 13-word runs" % len(bank), flush=True)
    return bank


def pieces(ids, piece, minimum):
    for start in range(0, len(ids), piece):
        chunk = ids[start:start + piece]
        if len(chunk) >= minimum:
            yield start, chunk


def hf_documents(repo, config, field):
    from datasets import load_dataset

    for index, row in enumerate(load_dataset(repo, config, split="train", streaming=True)):
        text = row.get(field) or ""
        if text.strip():
            yield str(row.get("id") or index), text


def code_documents(tokenizer):
    """Raw Python files from the local store, already tokenized at this vocabulary."""
    index = np.load(CODE_STORE / "train.idx.npy")
    stream = np.memmap(CODE_STORE / "train.bin", dtype=np.uint32, mode="r")
    # Walk the store from its far end, away from the prefix earlier runs trained on.
    for number in range(len(index) - 2, 0, -1):
        ids = np.asarray(stream[index[number]:index[number + 1]], dtype=np.int64).tolist()
        yield "code-%d" % number, ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-tokens", type=int, default=600_000)
    parser.add_argument("--scale", type=float, default=1.0,
                        help="multiplies every source's token budget")
    parser.add_argument("--piece", type=int, default=1024)
    parser.add_argument("--minimum", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--skip-documents", type=int, default=0,
                        help="skip this many documents of each source, so a larger build "
                             "can continue past a pilot instead of repeating it")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TEACHER)
    scored, related = build_banks()
    wiki = wikitext_bank()
    seen = set()
    report = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def take(label, domain, documents, budget, tokenized):
        stats = dict(kept=0, tokens=0, eval=0, scored=0, related=0, wikitext=0, duplicate=0,
                     documents=0)
        for doc_number, (doc_id, content) in enumerate(documents):
            if doc_number < args.skip_documents:
                continue
            if stats["tokens"] >= budget:
                break
            stats["documents"] += 1
            ids = content if tokenized else tokenizer(content, add_special_tokens=False)["input_ids"]
            for start, chunk in pieces(ids, args.piece, args.minimum):
                text = tokenizer.decode(chunk)
                key = fingerprint(text)
                if key in seen:
                    stats["duplicate"] += 1
                    continue
                grams = shingles(text)
                if grams & scored:
                    stats["scored"] += 1
                    continue
                if grams & related:
                    stats["related"] += 1
                    continue
                if shingles(text, (13,)) & wiki:
                    stats["wikitext"] += 1
                    continue
                seen.add(key)
                split = "eval" if stats["kept"] % args.eval_every == 0 else "train"
                out.write(json.dumps({"doc_id": "%s:%s:%d" % (label, doc_id, start),
                                      "input_ids": [int(t) for t in chunk], "split": split,
                                      "source": label, "domain": domain,
                                      "tokens": len(chunk)}) + "\n")
                stats["kept"] += 1
                stats["tokens"] += len(chunk)
                if split == "eval":
                    stats["eval"] += 1
                if stats["tokens"] >= budget:
                    break
        print("%-12s %s" % (label, json.dumps(stats)), flush=True)
        report[label] = stats

    with open(args.output, "w", encoding="utf-8") as out:
        for label, repo, config, field, budget, domain in SOURCES:
            take(label, domain, hf_documents(repo, config, field), int(budget * args.scale), False)
        take("code", "code", code_documents(tokenizer), int(args.code_tokens * args.scale), True)

    total = sum(s["tokens"] for s in report.values())
    print("total %d tokens in %d pieces -> %s"
          % (total, sum(s["kept"] for s in report.values()), args.output))
    args.output.with_suffix(".report.json").write_text(json.dumps(dict(
        sources=report, piece=args.piece, minimum=args.minimum, total_tokens=total,
        screened_against=["mmlu test", "arc-challenge test", "arc-easy test",
                          "mmlu/arc unscored splits", "wikitext-103 test"]), indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
