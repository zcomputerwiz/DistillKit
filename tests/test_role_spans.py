"""Half of what the independent screen measures is prompt, so it has to be split.

The held-out corpus is chat-formatted. Measured over the 384 screening documents:
median 180 tokens of near-identical system boilerplate before the assistant turn,
48.6% of the average 512-token window spent on system and user, and 19 documents whose
window never reaches the assistant at all. On top of that, 100% of assistant turns open
with an empty `<think></think>` block, so even the content half starts with a constant.

`role_spans` partitions a document's tokens so the same forward pass can be reported
per role. These pin the partition itself, which is the part that can silently be off by
a token per turn and never be noticed.
"""

import pytest

from distillkit.independent_eval import ROLES, role_spans


def offsets_for(text, pieces):
    """Character offsets for a hand-written tokenisation, plus a special token."""
    spans, cursor = [], 0
    for piece in pieces:
        index = text.index(piece, cursor)
        spans.append((index, index + len(piece)))
        cursor = index + len(piece)
    return spans


def flatten(spans):
    return {role: [t for low, high in ranges for t in range(low, high)]
            for role, ranges in spans.items()}


def test_every_token_lands_in_exactly_one_role():
    text = "<|im_start|>system\nBe good.<|im_start|>user\nHi<|im_start|>assistant\nHello"
    pieces = ["<|im_start|>", "system", "\n", "Be", " good", ".",
              "<|im_start|>", "user", "\n", "Hi",
              "<|im_start|>", "assistant", "\n", "Hello"]
    assigned = flatten(role_spans(text, offsets_for(text, pieces)))
    everything = [t for tokens in assigned.values() for t in tokens]
    assert sorted(everything) == list(range(len(pieces)))
    assert len(everything) == len(set(everything)), "a token was counted under two roles"
    assert set(assigned) <= set(ROLES)


def test_the_markers_are_template_and_the_bodies_are_their_role():
    text = "<|im_start|>system\nBe good.<|im_start|>user\nHi"
    pieces = ["<|im_start|>", "system", "\n", "Be", " good", ".",
              "<|im_start|>", "user", "\n", "Hi"]
    assigned = flatten(role_spans(text, offsets_for(text, pieces)))
    assert assigned["system"] == [3, 4, 5]
    assert assigned["user"] == [9]
    assert assigned["template"] == [0, 1, 2, 6, 7, 8]


def test_an_empty_think_block_is_template_not_assistant():
    """Present in 100% of this corpus's assistant turns; counting it as content would
    give every model a free constant to predict."""
    text = "<|im_start|>assistant\n<think>\n\n</think>\n\nReal answer"
    pieces = ["<|im_start|>", "assistant", "\n", "<think>", "\n\n", "</think>", "\n\n",
              "Real", " answer"]
    assigned = flatten(role_spans(text, offsets_for(text, pieces)))
    assert assigned["assistant"] == [7, 8]
    assert set(assigned["template"]) == {0, 1, 2, 3, 4, 5, 6}


def test_a_think_block_with_content_stays_assistant():
    """Only the empty one is an artifact. Real reasoning is what the model produced."""
    text = "<|im_start|>assistant\n<think>\nBecause\n</think>\n\nSo"
    pieces = ["<|im_start|>", "assistant", "\n", "<think>", "\n", "Because", "\n",
              "</think>", "\n\n", "So"]
    assigned = flatten(role_spans(text, offsets_for(text, pieces)))
    assert assigned["assistant"] == [3, 4, 5, 6, 7, 8, 9]
    assert assigned["template"] == [0, 1, 2]


def test_special_tokens_with_empty_offsets_are_skipped():
    """A fast tokenizer reports (0, 0) for tokens that consume no characters; charging
    them to whichever role starts at character zero would be arbitrary."""
    text = "<|im_start|>user\nHi"
    offsets = [(0, 0)] + offsets_for(text, ["<|im_start|>", "user", "\n", "Hi"])
    assigned = flatten(role_spans(text, offsets))
    assert 0 not in [t for tokens in assigned.values() for t in tokens]
    assert assigned["user"] == [4]


def test_truncation_keeps_the_spans_inside_the_window():
    """`prepare` passes only the offsets of the kept window; a span past its end would
    index a token that was never scored."""
    text = "<|im_start|>system\nBe good.<|im_start|>assistant\nHello there"
    pieces = ["<|im_start|>", "system", "\n", "Be", " good", ".",
              "<|im_start|>", "assistant", "\n", "Hello", " there"]
    window = offsets_for(text, pieces)[:8]
    spans = role_spans(text, window)
    assert max(high for ranges in spans.values() for _, high in ranges) <= len(window)
    assert "assistant" not in spans, "the window stops before any assistant content"


def test_plain_text_without_turns_is_all_assistant():
    """A corpus that is not chat-formatted should not silently score as zero tokens."""
    text = "Just some prose."
    spans = role_spans(text, offsets_for(text, ["Just", " some", " prose", "."]))
    assert spans == {"assistant": [[0, 4]]}


@pytest.mark.parametrize("role", ["tool", "developer"])
def test_an_unknown_role_is_counted_as_template(role):
    """New roles must not vanish from the accounting, and must not be silently mixed
    into assistant either."""
    text = f"<|im_start|>{role}\nPayload<|im_start|>assistant\nOk"
    pieces = ["<|im_start|>", role, "\n", "Payload", "<|im_start|>", "assistant", "\n", "Ok"]
    assigned = flatten(role_spans(text, offsets_for(text, pieces)))
    assert assigned["assistant"] == [7]
    assert 3 in assigned["template"]
