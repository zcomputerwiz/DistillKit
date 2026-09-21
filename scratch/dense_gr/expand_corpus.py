"""Render extra chat and code SFT corpora into the capture JSONL, screened for overlap.

The 5M-token capture corpus is a single dataset's ``sft_balanced`` config: 1,440 code
documents, 767 instruction documents, and a long tail of benchmark-derived QA. Two gaps
are worth filling. Plain multi-turn chat is nearly absent -- almost every row is one
instruction and one answer -- and the code is Evol-Code and CodeAlpaca, short synthetic
snippets rather than the file-scale code the model is asked about in practice.

That corpus also carries 145 ARC documents, 25 of which turned out to be questions the
ARC screen scores. So this screens rather than trusting the upstream split: a mixture
assembled from public SFT sets contains benchmark rows, and "usually drawn from train"
is not a check.

The screen is n-gram containment, the standard decontamination test. A document is
dropped when it shares an 8- or 13-word run with any benchmark question. Questions
shorter than 8 words contribute nothing, because a run that short is not evidence.
Choices and answers are deliberately not in the bank: they are short, formulaic, and
shared between unrelated questions, so they would drop documents that are merely about
the same subject.

Documents are also deduplicated against the existing corpus, since two of the four
sources here are already partly represented in it.

Rendering follows ``distillkit.prepare_corpus``: the teacher's own chat template, whole
documents one per row, unpadded. Long documents are cut at the capture's
``sequence_length`` (4096) because capture truncates there anyway, and a 30k-token row
would otherwise spend the whole budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from distillkit.prepare_corpus import render_messages  # noqa: E402

WORDS = re.compile(r"[a-z0-9]+")
# The two shingle widths. 13 is the usual decontamination width; 8 catches the many
# benchmark questions that are shorter than 13 words and would otherwise be invisible.
WIDTHS = (13, 8)
SHORTEST = 8


def _magicoder(row):
    return [{"role": "user", "content": row["problem"]},
            {"role": "assistant", "content": row["solution"]}]


def _self_oss(row):
    return [{"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["response"]}]


def _messages(row):
    return row["messages"]


SOURCES = [
    # repo, split, adapter, domain, label
    ("ise-uiuc/Magicoder-OSS-Instruct-75K", "train", _magicoder, "code", "Magicoder-OSS"),
    ("bigcode/self-oss-instruct-sc2-exec-filter-50k", "train", _self_oss, "code",
     "self-oss-instruct"),
    ("allenai/tulu-3-sft-mixture", "train", _messages, "instruction", "tulu-3-sft"),
    ("HuggingFaceH4/ultrachat_200k", "train_sft", _messages, "chat", "ultrachat"),
]

# The full question banks rather than the 512 sampled, so a later resample stays clean
# too. Split by whether the eval actually scores that split: a document overlapping
# MMLU's test set is contamination, one overlapping ARC's train set is only a
# near-duplicate of the benchmark's distribution. Both are dropped -- together they are
# a couple of percent of any corpus -- but they are counted apart, because reading the
# second as contamination overstates the first.
SCORED = [("cais/mmlu", "all", ("test",)),
          ("allenai/ai2_arc", "ARC-Challenge", ("test",)),
          ("allenai/ai2_arc", "ARC-Easy", ("test",))]
RELATED = [("cais/mmlu", "all", ("validation", "dev")),
           ("allenai/ai2_arc", "ARC-Challenge", ("validation", "train")),
           ("allenai/ai2_arc", "ARC-Easy", ("validation", "train"))]


def shingles(text, widths=WIDTHS):
    """The word n-grams of `text` at each width.

    Kept as strings rather than `hash()` values: the join has to happen either way, so
    the strings cost only memory, and `hash()` is salted per process, which would make
    a saved bank silently match nothing on the next run.
    """
    words = WORDS.findall(text.lower())
    out = set()
    for width in widths:
        if len(words) < width:
            continue
        for start in range(len(words) - width + 1):
            out.add(" ".join(words[start:start + width]))
    return out


def question_shingles(text):
    """The shingles a benchmark question contributes to the bank.

    A question of 13 words or more contributes its 13-grams; a shorter one contributes
    its 8-grams, which is the narrowest run still specific enough to mean something.
    Anything under 8 words contributes nothing.
    """
    words = WORDS.findall(text.lower())
    if len(words) >= 13:
        return shingles(text, (13,))
    if len(words) >= SHORTEST:
        return shingles(text, (SHORTEST,))
    return set()


def build_bank(banks):
    from datasets import load_dataset

    bank, questions, skipped = set(), 0, 0
    for repo, config, splits in banks:
        for split in splits:
            try:
                rows = load_dataset(repo, config, split=split)
            except Exception as exc:                      # a split a release dropped
                print("  %s/%s %s unavailable (%s)" % (repo, config, split, exc))
                continue
            for text in rows["question"]:
                grams = question_shingles(text)
                if grams:
                    bank |= grams
                else:
                    skipped += 1
                questions += 1
            print("  %-22s %-14s %-11s %7d questions" % (repo, config, split, len(rows)))
    print("  -> %d questions, %d shingles, %d too short to screen"
          % (questions, len(bank), skipped))
    return bank


def build_banks():
    print("scored splits (overlap here is contamination)")
    scored = build_bank(SCORED)
    print("unscored splits (overlap here is a near-duplicate, not contamination)")
    related = build_bank(RELATED) - scored
    return scored, related


def fingerprint(text):
    """A dedup key robust to whitespace and punctuation.

    Hashes the whole normalised word sequence, not a prefix of it. A prefix looks
    cheaper and is worthless here: these documents open with a chat template and often
    a dataset-wide system prompt, so the first few hundred characters are shared by
    every row from a given source. Truncating at 400 characters collapsed a
    5,598-document corpus to 11 distinct keys.
    """
    return hashlib.sha1(" ".join(WORDS.findall(text.lower())).encode()).hexdigest()


def existing_fingerprints(path):
    seen = set()
    if not path or not Path(path).exists():
        return seen
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            seen.add(fingerprint(json.loads(line)["text"]))
    return seen


def iter_source(repo, split, adapter, domain, label, tokenizer, max_tokens):
    from datasets import load_dataset

    rows = load_dataset(repo, split=split, streaming=True)
    for index, row in enumerate(rows):
        try:
            messages = adapter(row)
        except (KeyError, TypeError):
            continue
        text = render_messages(messages or [], tokenizer)
        if not text:
            continue
        ids = tokenizer(text)["input_ids"]
        if len(ids) < 16:
            continue
        if len(ids) > max_tokens:
            text = tokenizer.decode(ids[:max_tokens])
            ids = ids[:max_tokens]
        yield {"doc_id": "%s:%s" % (label, row.get("id") or row.get("sha1") or index),
               "text": text, "split": "train", "source": label, "domain": domain,
               "subsource": row.get("source"), "tokens": len(ids)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="D:/DeepThought/Projects/HybridModel/teacher-hf")
    parser.add_argument("--output", type=Path,
                        default=Path("../capture-data/expand.jsonl"))
    parser.add_argument("--report", type=Path,
                        default=Path("../capture-data/expand-screen.json"))
    parser.add_argument("--against", type=Path,
                        default=Path("../capture-data/run5m.jsonl"),
                        help="corpus to deduplicate against")
    parser.add_argument("--tokens-per-source", type=int, default=1_250_000)
    parser.add_argument("--max-doc-tokens", type=int, default=4096,
                        help="the capture's sequence_length; longer rows are cut there")
    parser.add_argument("--screen-only", type=Path,
                        help="audit an existing JSONL for benchmark overlap and stop")
    args = parser.parse_args()

    print("building the benchmark bank")
    scored, related = build_banks()

    def verdict(text):
        """"scored", "related" or None -- which bank, if either, this text overlaps."""
        grams = shingles(text)
        if grams & scored:
            return "scored"
        return "related" if grams & related else None

    if args.screen_only:
        counts, total = {}, 0
        with open(args.screen_only, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                total += 1
                hit = verdict(row["text"])
                if hit:
                    key = (hit, row.get("source"))
                    counts[key] = counts.get(key, 0) + 1
        for name in ("scored", "related"):
            share = sum(v for (kind, _), v in counts.items() if kind == name)
            print("\n%s: %d of %d documents overlap a %s split (%.2f%%)"
                  % (args.screen_only, share, total, name, 100.0 * share / max(1, total)))
            for (kind, source), count in sorted(counts.items(), key=lambda kv: -kv[1]):
                if kind == name:
                    print("  %6d  %s" % (count, source))
        return 0

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    seen = existing_fingerprints(args.against)
    print("deduplicating against %d documents in %s" % (len(seen), args.against))

    streams = [(label, domain,
                iter_source(repo, split, adapter, domain, label, tokenizer,
                            args.max_doc_tokens))
               for repo, split, adapter, domain, label in SOURCES]
    stats = {label: {"kept": 0, "tokens": 0, "scored": 0, "related": 0, "duplicate": 0}
             for label, _, _ in streams}
    dropped = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.output, "w", encoding="utf-8") as out:
        active = list(streams)
        while active:
            for entry in list(active):
                label, _, stream = entry
                record = stats[label]
                if record["tokens"] >= args.tokens_per_source:
                    active.remove(entry)
                    continue
                # One document per source per turn, so a run cut short stays balanced.
                for row in stream:
                    key = fingerprint(row["text"])
                    if key in seen:
                        record["duplicate"] += 1
                        continue
                    hit = verdict(row["text"])
                    if hit:
                        record[hit] += 1
                        dropped.append({"doc_id": row["doc_id"], "bank": hit,
                                        "source": label,
                                        "subsource": row.get("subsource")})
                        continue
                    seen.add(key)
                    record["kept"] += 1
                    record["tokens"] += row["tokens"]
                    out.write(json.dumps(row) + "\n")
                    written += 1
                    if written % 500 == 0:
                        print("  %d documents, %d tokens"
                              % (written, sum(s["tokens"] for s in stats.values())),
                              flush=True)
                    break
                else:
                    active.remove(entry)

    columns = ("kept", "tokens", "scored", "related", "duplicate")
    print("\n%-20s %8s %12s %8s %9s %11s" % (("source",) + columns))
    for label, record in stats.items():
        print("%-20s %8d %12d %8d %9d %11d"
              % ((label,) + tuple(record[c] for c in columns)))
    print("%-20s %8d %12d %8d %9d %11d"
          % (("total",) + tuple(sum(s[c] for s in stats.values()) for c in columns)))
    print("\nwrote %s" % args.output)

    args.report.write_text(json.dumps(
        {"sources": stats, "widths": list(WIDTHS),
         "scored_shingles": len(scored), "related_shingles": len(related),
         "scored_banks": [[r, c, list(s)] for r, c, s in SCORED],
         "related_banks": [[r, c, list(s)] for r, c, s in RELATED],
         "deduplicated_against": str(args.against),
         "max_doc_tokens": args.max_doc_tokens,
         "dropped": dropped}, indent=2), encoding="utf-8")
    print("wrote %s" % args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
