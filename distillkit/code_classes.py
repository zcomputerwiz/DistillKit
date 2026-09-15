"""Code-aware target-class decomposition, refining the historical one rather than replacing it.

The earlier experiments reported five classes -- content, newline, whitespace, punctuation,
control -- and every result in this programme is quoted in them. Python needs finer
resolution than that: "punctuation" pools the delimiters that carry Python's grammar with
the operators that carry its semantics, and "content" pools identifiers with keywords.

So this is a *refinement*, not a second opinion. Every code class maps to exactly one
historical class (:data:`HISTORICAL`), which means the two views are arithmetically
consistent by construction -- the historical numbers are sums of code-class numbers, and
cannot disagree with them. Reporting two independently-derived classifications of the same
tokens would invite exactly the kind of quiet contradiction that is impossible to notice
in a table.

Classification is on **tokenizer-visible text only**: the token's own decoded string, with
a leading space stripped. It is deliberately not parse-aware. A ``.`` is a delimiter here
whether it is attribute access or part of a float, because deciding otherwise would
require knowing the surrounding source, and a class that depends on context is not a
property of the target the model predicts. Where tokenization makes a distinction
ambiguous, the coarser answer wins.

One boundary is worth stating because Python disagrees with intuition: ``and``, ``or``,
``not``, ``in`` and ``is`` read as operators but are *keywords* in the language grammar,
and they are classified as keywords here on the authority of ``keyword.kwlist`` rather
than by hand. That is the deterministic choice; the alternative is a curated list that
drifts from the language.
"""

from __future__ import annotations

import keyword

__all__ = [
    "CODE_CLASSES",
    "DELIMITERS",
    "HISTORICAL",
    "HISTORICAL_CLASSES",
    "OPERATORS",
    "build_code_classes",
    "code_class_of",
    "historical_of",
]

#: Python's delimiters: the characters the grammar uses for structure. From the language
#: reference, minus the augmented-assignment forms, which are operators below.
DELIMITERS = frozenset({
    "(", ")", "[", "]", "{", "}", ",", ":", ".", ";", "@", "=", "->",
})

#: Operators, including the augmented assignments. ``=`` is a delimiter in Python's own
#: grammar but is listed in both there; it is resolved to delimiter, above, since plain
#: assignment is structural in the same sense a colon is.
OPERATORS = frozenset({
    "+", "-", "*", "/", "//", "%", "**", "@",
    "<<", ">>", "&", "|", "^", "~", ":=",
    "<", ">", "<=", ">=", "==", "!=",
    "+=", "-=", "*=", "/=", "//=", "%=", "**=",
    ">>=", "<<=", "&=", "|=", "^=", "@=",
})

CODE_CLASSES = ("content", "keyword", "operator", "delimiter", "newline",
                "whitespace", "other_punct", "control")

HISTORICAL_CLASSES = ("content", "newline", "whitespace", "punctuation", "control")

#: Each code class rolls up into exactly one historical class, so the five-way numbers
#: quoted by every earlier experiment are sums of these and cannot drift from them.
HISTORICAL = {
    "content": "content",
    "keyword": "content",
    "operator": "punctuation",
    "delimiter": "punctuation",
    "other_punct": "punctuation",
    "newline": "newline",
    "whitespace": "whitespace",
    "control": "control",
}

_KEYWORDS = frozenset(keyword.kwlist) | frozenset(keyword.softkwlist)


def code_class_of(text: str, is_special: bool) -> str:
    """Classify one token from its decoded text. Ordered so the rules do not overlap."""
    if is_special:
        return "control"
    if text == "":
        return "control"
    if "\n" in text or "\r" in text:
        return "newline"
    if text.isspace():
        return "whitespace"
    stripped = text.strip()
    if stripped in _KEYWORDS:
        return "keyword"
    # ``@`` is both a decorator delimiter and the matrix-multiply operator; decorators are
    # overwhelmingly the commoner use in ordinary Python, so it resolves to delimiter.
    if stripped in DELIMITERS:
        return "delimiter"
    if stripped in OPERATORS:
        return "operator"
    if not any(character.isalnum() for character in stripped):
        return "other_punct"
    return "content"


def historical_of(code_class: str) -> str:
    return HISTORICAL[code_class]


def build_code_classes(tokenizer, vocab_size: int) -> list[str]:
    """Classify every id in the model's vocabulary, once.

    ``vocab_size`` is the model's embedding row count, which exceeds the tokenizer's
    vocabulary -- the rows beyond it are padding to a hardware-friendly multiple and
    decode to nothing. They are classified ``control`` so that if one is ever predicted
    or scored it lands somewhere visible rather than being silently counted as content.
    """
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    special.update(int(key) for key in (getattr(tokenizer, "added_tokens_decoder", {}) or {}))

    tokens = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
    classes = []
    for index, token in enumerate(tokens):
        if token is None:
            classes.append("control")
            continue
        text = tokenizer.convert_tokens_to_string([token])
        classes.append(code_class_of(text, index in special))
    return classes
