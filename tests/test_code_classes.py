"""Tests for the code-aware target-class decomposition.

The classification decides which tokens a reported number is about, so an error here does
not produce a wrong answer -- it produces a right-looking answer to a different question.
Two properties carry that weight and are pinned hardest: every code class rolls up into
exactly one historical class, so the five-way view quoted by every earlier experiment is
arithmetically a sum of the eight-way view and cannot contradict it; and classification
depends only on the token's own text, so it is a property of the target rather than of
its context.
"""

import keyword

import pytest

from distillkit.code_classes import (CODE_CLASSES, DELIMITERS, HISTORICAL,
                                     HISTORICAL_CLASSES, OPERATORS, build_code_classes,
                                     code_class_of, historical_of)


class TestRollup:
    def test_every_code_class_maps_to_exactly_one_historical_class(self):
        assert set(HISTORICAL) == set(CODE_CLASSES)
        assert set(HISTORICAL.values()) <= set(HISTORICAL_CLASSES)

    def test_the_historical_view_is_a_partition_of_the_code_view(self):
        """No historical class is left without a source and none is double-counted."""
        covered = {}
        for code in CODE_CLASSES:
            covered.setdefault(HISTORICAL[code], set()).add(code)
        assert covered["content"] == {"content", "keyword"}
        assert covered["punctuation"] == {"operator", "delimiter", "other_punct"}
        assert covered["newline"] == {"newline"}
        assert covered["whitespace"] == {"whitespace"}
        assert covered["control"] == {"control"}

    def test_keywords_are_content_not_punctuation(self):
        """``and`` reads like an operator but is content: it carries meaning, not layout."""
        assert historical_of("keyword") == "content"
        assert historical_of("operator") == "punctuation"


class TestClassification:
    @pytest.mark.parametrize("text,expected", [
        ("def", "keyword"), (" return", "keyword"), ("class", "keyword"),
        ("and", "keyword"), ("not", "keyword"), ("in", "keyword"), ("is", "keyword"),
        ("match", "keyword"),                       # soft keyword
        ("(", "delimiter"), (" [", "delimiter"), (":", "delimiter"), (",", "delimiter"),
        (".", "delimiter"), ("=", "delimiter"), ("->", "delimiter"),
        ("+", "operator"), ("**", "operator"), ("//", "operator"), ("==", "operator"),
        ("!=", "operator"), (">=", "operator"), ("+=", "operator"), (":=", "operator"),
        ("<<", "operator"), ("^", "operator"), ("~", "operator"),
        ("\n", "newline"), ("\n\n", "newline"), ("\r\n", "newline"), ("    \n", "newline"),
        ("    ", "whitespace"), (" ", "whitespace"), ("\t", "whitespace"),
        ("#", "other_punct"), ('"""', "other_punct"), ("!", "other_punct"),
        ("foo", "content"), (" self", "content"), ("42", "content"), ("x1", "content"),
        ("_private", "content"),
    ])
    def test_representative_tokens(self, text, expected):
        assert code_class_of(text, False) == expected

    def test_special_tokens_are_control_whatever_they_decode_to(self):
        assert code_class_of("def", True) == "control"
        assert code_class_of("<|im_start|>", True) == "control"
        assert code_class_of("", False) == "control"

    def test_newline_beats_whitespace(self):
        """A token of spaces *and* a newline is a newline; both would otherwise match."""
        assert code_class_of("\n   ", False) == "newline"
        assert code_class_of("   \n", False) == "newline"

    def test_leading_space_does_not_change_the_class(self):
        """BPE marks word starts with a leading space; ``if`` and `` if`` are one keyword."""
        for text in ("if", "def", "+", "(", "foo"):
            assert code_class_of(text, False) == code_class_of(" " + text, False)

    def test_classification_uses_only_the_token_text(self):
        """A class that depended on context would not be a property of the target.

        ``.`` is a delimiter here whether it is attribute access or part of ``3.14``,
        because the model predicts the token, not the parse.
        """
        assert code_class_of(".", False) == "delimiter"

    def test_every_python_keyword_is_reachable(self):
        for word in keyword.kwlist:
            assert code_class_of(word, False) == "keyword"

    def test_operator_and_delimiter_sets_are_disjoint_except_by_precedence(self):
        """``=`` and ``@`` are in both of Python's own lists; the tie is broken once."""
        overlap = DELIMITERS & OPERATORS
        assert overlap == {"@"}
        assert code_class_of("@", False) == "delimiter"
        assert code_class_of("=", False) == "delimiter"

    def test_result_is_always_a_declared_class(self):
        for text in ("", " ", "\n", "def", "@@@", "é", "0x1f", "->", "\x00"):
            assert code_class_of(text, False) in CODE_CLASSES


class TestVocabularyTable:
    class FakeTokenizer:
        """Ids 0-4 are real tokens, 5 is special, 6 and 7 are embedding padding."""

        all_special_ids = [5]
        added_tokens_decoder = {5: object()}

        def convert_ids_to_tokens(self, ids):
            table = {0: "def", 1: "Ġfoo", 2: "(", 3: "Ċ", 4: "+", 5: "<|end|>"}
            return [table.get(i) for i in ids]

        def convert_tokens_to_string(self, tokens):
            return tokens[0].replace("Ġ", " ").replace("Ċ", "\n")

    def test_padding_rows_beyond_the_tokenizer_are_control_not_content(self):
        """Rows past the tokenizer decode to nothing and must not land in content.

        The model's embedding is padded to a hardware-friendly multiple, so ids exist
        that no tokenizer entry corresponds to. If one were ever scored, silently
        counting it as content would move a content number.
        """
        classes = build_code_classes(self.FakeTokenizer(), 8)
        assert classes == ["keyword", "content", "delimiter", "newline", "operator",
                           "control", "control", "control"]

    def test_table_length_matches_the_embedding_not_the_tokenizer(self):
        assert len(build_code_classes(self.FakeTokenizer(), 8)) == 8
        assert len(build_code_classes(self.FakeTokenizer(), 6)) == 6
