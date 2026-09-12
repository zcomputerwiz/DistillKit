"""Tokenizer-derived token classes, shared by every analysis that splits a corpus.

The project's `LAYOUT_TOKEN_IDS` named three ids -- `\n` and the two think tags -- and
that is not the whitespace vocabulary. This corpus's assistant targets use 35 to 41
whitespace-only types depending on the bundle, `\n\n` and eight distinct runs of spaces
among them, and scoring against the three-id mask put `\n\n` in the content column where
it was over half the apparent content gain (`scratch/ple_forensics/RESULTS.md`).

So classes are derived from the vocabulary, never hardcoded:

``whitespace``  decoded text is non-empty and entirely whitespace
``control``     the chat/protocol markers and the think tags
``punctuation`` decoded text is non-empty and entirely punctuation or symbols
``lexical``     everything else -- the content the model is actually for

`punctuation` is reported separately rather than folded into either side. A full stop is
not layout, but it is not what "content modelling" means either, and in the capacity
probe's content column `.` was the second largest single gainer.
"""

from __future__ import annotations

import unicodedata

import numpy as np

CLASSES = ("whitespace", "control", "punctuation", "lexical")

#: Chat protocol and reasoning markers. These are ids rather than a derived rule because
#: they are special tokens whose decoded form is their literal spelling; there is no
#: property of the string that distinguishes `<|im_end|>` from a word.
CONTROL_TOKEN_IDS = frozenset((248044, 248045, 248046, 248068, 248069))


def _is_punctuation(text: str) -> bool:
    # Unicode categories P* (punctuation) and S* (symbols). Byte-level BPE spells a
    # leading space as U+0120, so strip the marker before judging.
    stripped = text.replace("Ġ", "").replace("Ċ", "")
    if not stripped:
        return False
    return all(unicodedata.category(character)[0] in "PS" for character in stripped)


def classify(token_id: int, tokenizer) -> str:
    if int(token_id) in CONTROL_TOKEN_IDS:
        return "control"
    text = tokenizer.decode([int(token_id)])
    if text and not text.strip():
        return "whitespace"
    if _is_punctuation(text):
        return "punctuation"
    return "lexical"


def class_table(token_ids, tokenizer) -> dict[int, str]:
    """One decode per distinct id, not one per position."""
    return {int(token): classify(int(token), tokenizer)
            for token in np.unique(np.asarray(token_ids))}


def class_of(token_ids, tokenizer) -> np.ndarray:
    table = class_table(token_ids, tokenizer)
    return np.array([table[int(token)] for token in np.asarray(token_ids)])


def masks(token_ids, tokenizer) -> dict[str, np.ndarray]:
    labels = class_of(token_ids, tokenizer)
    return {name: labels == name for name in CLASSES}


def summarise(token_ids, tokenizer) -> str:
    labels = class_of(token_ids, tokenizer)
    lines = ["%-12s %8s %7s  %s" % ("class", "tokens", "share", "distinct types")]
    table = class_table(token_ids, tokenizer)
    for name in CLASSES:
        mask = labels == name
        types = sum(1 for value in table.values() if value == name)
        lines.append("%-12s %8d %6.1f%%  %d"
                     % (name, mask.sum(), 100 * mask.mean(), types))
    return "\n".join(lines)
