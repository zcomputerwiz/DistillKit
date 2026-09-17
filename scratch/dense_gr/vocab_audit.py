"""What is actually in the 248,320-entry vocabulary, and what Python uses of it."""
import sys
import unicodedata

import numpy as np
from transformers import AutoTokenizer

BASE = "D:/DeepThought/Projects/HybridModel/student-2b-hf"
VOCAB = 248_320

tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
added = tokenizer.get_added_vocab()
print("tokenizer.vocab_size %d   len(tokenizer) %d   added %d"
      % (tokenizer.vocab_size, len(tokenizer), len(added)))
print()
print("=== added / special tokens ===")
for name, index in sorted(added.items(), key=lambda kv: kv[1]):
    print("%8d  %s" % (index, name))

RANGES = [
    ("cjk", [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF)]),
    ("kana", [(0x3040, 0x30FF), (0x31F0, 0x31FF)]),
    ("hangul", [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)]),
    ("cyrillic", [(0x0400, 0x052F)]),
    ("greek", [(0x0370, 0x03FF)]),
    ("arabic", [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF)]),
    ("hebrew", [(0x0590, 0x05FF)]),
    ("devanagari", [(0x0900, 0x097F)]),
    ("thai", [(0x0E00, 0x0E7F)]),
    ("emoji_symbols", [(0x1F000, 0x1FAFF), (0x2600, 0x27BF), (0x2B00, 0x2BFF)]),
]


def classify(text):
    if text == "":
        return "empty"
    worst = "ascii"
    for character in text:
        point = ord(character)
        if point < 128:
            continue
        for name, spans in RANGES:
            if any(low <= point <= high for low, high in spans):
                return name
        if 0x80 <= point <= 0x024F:
            worst = "latin_extended"
        else:
            worst = "other_nonascii"
    return worst


print()
print("decoding %d ids..." % VOCAB, flush=True)
texts = tokenizer.batch_decode([[i] for i in range(VOCAB)])
special = set(added.values())
categories = []
for index, text in enumerate(texts):
    categories.append("special" if index in special else classify(text))
categories = np.array(categories)

tokens = np.memmap("scratch/code_training/tokens/train.bin", dtype=np.uint32, mode="r")
counts = np.bincount(np.asarray(tokens, dtype=np.int64), minlength=VOCAB)
total = int(counts.sum())

print()
print("%-16s %9s %8s %14s %10s" % ("category", "ids", "% vocab", "python tokens", "% corpus"))
rows = []
for name in sorted(set(categories.tolist())):
    mask = categories == name
    ids = int(mask.sum())
    used = int(counts[mask].sum())
    rows.append((used, name, ids, used))
for used, name, ids, _ in sorted(rows, reverse=True):
    print("%-16s %9d %7.2f%% %14d %9.4f%%"
          % (name, ids, 100 * ids / VOCAB, used, 100 * used / total))

nonlatin = np.isin(categories, ["cjk", "kana", "hangul", "cyrillic", "greek", "arabic",
                                "hebrew", "devanagari", "thai", "emoji_symbols",
                                "other_nonascii"])
print()
print("non-Latin-script ids: %d (%.1f%% of vocabulary), %.4f%% of Python corpus tokens"
      % (int(nonlatin.sum()), 100 * nonlatin.sum() / VOCAB,
         100 * counts[nonlatin].sum() / total))
keep = ~nonlatin
print("keeping ASCII + latin_extended + special: %d ids, %.4f%% coverage"
      % (int(keep.sum()), 100 * counts[keep].sum() / total))
unused = counts == 0
print("ids never used in 30.7M Python tokens: %d (%.1f%%)"
      % (int(unused.sum()), 100 * unused.sum() / VOCAB))
