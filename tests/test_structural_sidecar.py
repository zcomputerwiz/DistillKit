"""A post-hoc correction is only measurable if the thing it corrects cannot move.

The whole design rests on three mechanical facts: the backbone takes no gradient, the
sidecar touches structural logits and nothing else, and `strength = 0` is the stock model
exactly. The fourth is about interpretation rather than mechanism -- the fixed-code arm
has to be genuinely fixed, or "learned rows are unnecessary" would be a statement about
rows that quietly trained anyway.
"""

import pytest
import torch

from distillkit.experimental.ngram_hash import NGramHashConfig, NGramHasher
from distillkit.experimental.structural_sidecar import (
    MODES, StructuralSidecar, apply_structural_bias, structural_token_ids,
    wrong_context_rows)

VOCAB = 64
STRUCTURAL = (3, 5, 7, 11)


def hasher(vocab=VOCAB, base=97):
    return NGramHasher(NGramHashConfig(
        vocab_size=vocab, ngram_size=3, heads_per_ngram=1,
        ngram_vocab_size_base=base, eos_token_id=vocab - 1, seed=7))


def sidecar(mode="fixed", rows=512, code_dim=8, hidden=0, seed=3):
    return StructuralSidecar(rows=rows, code_dim=code_dim, structural=len(STRUCTURAL),
                             mode=mode, heads=2, hidden=hidden, seed=seed)


def ids(batch=2, length=9, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB - 1, (batch, length), generator=generator)


def structural_index():
    return torch.tensor(STRUCTURAL, dtype=torch.long)


# --- identity -----------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_a_fresh_sidecar_biases_nothing(mode):
    module = sidecar(mode)
    rows = hasher().row_indices(ids())
    assert torch.equal(module(rows), torch.zeros(2, 9, len(STRUCTURAL)))


def test_strength_zero_returns_the_logits_untouched():
    logits = torch.randn(2, 9, VOCAB)
    bias = torch.randn(2, 9, len(STRUCTURAL))
    assert apply_structural_bias(logits, bias, structural_index(), 0.0) is logits


def test_the_bias_lands_only_on_structural_columns():
    logits = torch.randn(2, 9, VOCAB)
    bias = torch.randn(2, 9, len(STRUCTURAL))
    index = structural_index()
    out = apply_structural_bias(logits, bias, index, 1.0)
    moved = (out != logits).any(dim=(0, 1)).nonzero().reshape(-1)
    assert set(moved.tolist()) <= set(STRUCTURAL)
    for position, token in enumerate(STRUCTURAL):
        assert torch.allclose(out[..., token], logits[..., token] + bias[..., position])


def test_strength_scales_the_correction():
    logits = torch.zeros(1, 4, VOCAB)
    bias = torch.ones(1, 4, len(STRUCTURAL))
    index = structural_index()
    half = apply_structural_bias(logits, bias, index, 0.5)
    full = apply_structural_bias(logits, bias, index, 1.0)
    assert torch.allclose(half[..., STRUCTURAL[0]], 0.5 * full[..., STRUCTURAL[0]])


def test_a_mismatched_bias_is_refused():
    logits = torch.randn(2, 9, VOCAB)
    with pytest.raises(ValueError, match="structural columns"):
        apply_structural_bias(logits, torch.randn(2, 9, 2), structural_index(), 1.0)
    with pytest.raises(ValueError, match="does not match logits"):
        apply_structural_bias(logits, torch.randn(2, 3, len(STRUCTURAL)),
                              structural_index(), 1.0)


# --- what trains and what does not ---------------------------------------------


def test_fixed_codes_are_buffers_and_never_train():
    module = sidecar("fixed")
    assert "codes" in dict(module.named_buffers())
    assert not any(name.startswith("codes") for name, _ in module.named_parameters())
    before = module.codes.clone()
    module(hasher().row_indices(ids())).sum().backward()
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.1, weight_decay=0.5)
    optimizer.step()
    assert torch.equal(module.codes, before)


def test_the_learned_table_does_train():
    module = sidecar("table")
    assert "codes" in dict(module.named_parameters())
    # The decoder starts at zero, so the rows take no gradient on the first step; give
    # the output layer a nudge first, exactly as a real run's first update would.
    module.decoder[-1].weight.data.normal_(0, 0.1)
    module(hasher().row_indices(ids())).sum().backward()
    assert module.codes.grad is not None and torch.any(module.codes.grad != 0)


def test_the_unaddressed_arm_has_no_table_at_all():
    module = sidecar("none")
    assert module.codes is None
    assert "constant" in dict(module.named_parameters())
    report = module.parameter_report()
    assert report["frozen_code_entries"] == 0


def test_fixed_codes_are_reproducible_from_the_seed():
    assert torch.equal(sidecar("fixed", seed=11).codes, sidecar("fixed", seed=11).codes)
    assert not torch.equal(sidecar("fixed", seed=11).codes,
                           sidecar("fixed", seed=12).codes)


def test_the_fixed_arm_is_far_smaller_than_the_learned_one():
    fixed = sidecar("fixed").parameter_report()
    table = sidecar("table").parameter_report()
    assert fixed["trainable_parameters"] < table["trainable_parameters"]


# --- addressing ----------------------------------------------------------------


def test_the_same_history_hashes_the_same_way():
    tokens = ids()
    assert torch.equal(hasher().row_indices(tokens), hasher().row_indices(tokens.clone()))


def test_an_eos_resets_the_history():
    """The historical semantics: context does not cross a document boundary."""
    eos = VOCAB - 1
    left = torch.tensor([[5, 6, eos, 9, 4]])
    right = torch.tensor([[1, 2, eos, 9, 4]])
    rows = hasher().row_indices(left)
    other = hasher().row_indices(right)
    assert torch.equal(rows[:, 3:], other[:, 3:]), "post-EOS rows must not see the prefix"
    assert not torch.equal(rows[:, :2], other[:, :2])


def test_wrong_context_actually_moves_the_addresses():
    rows = hasher().row_indices(ids(length=16))
    wrong = wrong_context_rows(rows, shift=7)
    assert wrong.shape == rows.shape
    assert not torch.equal(wrong, rows)
    with pytest.raises(ValueError, match="has to move"):
        wrong_context_rows(rows, shift=0)


def test_wrong_context_changes_what_a_trained_sidecar_predicts():
    module = sidecar("fixed")
    module.decoder[-1].weight.data.normal_(0, 0.5)
    rows = hasher().row_indices(ids(length=16))
    assert not torch.equal(module(rows), module(wrong_context_rows(rows)))


def test_the_unaddressed_arm_ignores_the_addresses():
    """Which is the point of it: no local identity, matched decoder."""
    module = sidecar("none")
    module.decoder[-1].weight.data.normal_(0, 0.5)
    rows = hasher().row_indices(ids(length=16))
    assert torch.equal(module(rows), module(wrong_context_rows(rows)))


# --- guards ---------------------------------------------------------------------


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="unknown mode"):
        StructuralSidecar(rows=8, code_dim=4, structural=3, mode="vibes")


def test_a_head_count_mismatch_is_refused():
    module = sidecar("fixed")
    with pytest.raises(ValueError, match="expected 2 heads"):
        module(torch.zeros(1, 4, 3, dtype=torch.long))


def test_structural_ids_come_from_the_class_splitter():
    classes = ["content", "punctuation", "content", "layout", "control"]
    assert structural_token_ids(classes).tolist() == [1, 3, 4]
    with pytest.raises(ValueError, match="no structural tokens"):
        structural_token_ids(["content", "content"])
