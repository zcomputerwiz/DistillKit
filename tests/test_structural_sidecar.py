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
from distillkit.experimental.ngram_hash import splitmix64
from distillkit.experimental.structural_sidecar import (
    MODES, FactorizedSidecar, StructuralSidecar, WhitespaceBias,
    apply_structural_bias, signed_hash_features, splitmix64_tensor,
    structural_token_ids, wrong_context_rows)

MASK64 = (1 << 64) - 1

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


# --- table-free codes -----------------------------------------------------------
#
# The previous experiment found that learned rows were unnecessary once decoder capacity
# was matched, which leaves an obvious question: is a stored random basis necessary
# either? The direct mode answers it by slicing signed features out of one mixed hash
# word. Everything below exists because "no table" has to mean no table -- not a table
# that is allocated somewhere less visible -- and because a hand-vectorised mixer is
# exactly the kind of code that is subtly wrong and still looks random.


def test_the_vectorised_mixer_matches_the_scalar_reference():
    """Bit-identity, not distributional similarity. torch has no unsigned int64."""
    values = [0, 1, 2, 12345, 2 ** 40, 2 ** 62, 131199, -5, -(2 ** 62)]
    mixed = splitmix64_tensor(torch.tensor(values, dtype=torch.int64))
    for value, result in zip(values, mixed.tolist()):
        assert splitmix64(value & MASK64) == (result & MASK64), value


def test_direct_features_are_signed_and_normalised():
    features = signed_hash_features(torch.arange(64).reshape(1, 32, 2), 32, seed=7)
    assert features.shape == (1, 32, 2, 32)
    # float32, so compare against the value the construction actually produces rather
    # than against Python's float64 reciprocal square root.
    magnitude = torch.tensor(1.0 / 32 ** 0.5)
    assert torch.equal(features.abs(), magnitude.expand_as(features))
    assert torch.allclose(features.norm(dim=-1), torch.ones(1, 32, 2), atol=1e-5)


def test_direct_features_are_deterministic():
    rows = torch.arange(128).reshape(1, 64, 2)
    assert torch.equal(signed_hash_features(rows, 32, 7),
                       signed_hash_features(rows.clone(), 32, 7))
    assert not torch.equal(signed_hash_features(rows, 32, 7),
                           signed_hash_features(rows, 32, 8))


def test_direct_features_decorrelate_distinct_rows():
    """A mixer that collapsed nearby ids would look fine and carry no identity."""
    rows = torch.arange(4096, dtype=torch.int64)
    features = signed_hash_features(rows.reshape(1, 2048, 2), 32, seed=7)
    flat = features.reshape(-1, 32)
    assert len({tuple(row) for row in flat.tolist()}) > 4000
    gram = (flat @ flat.T) * 32
    off = gram - torch.diag(torch.diagonal(gram))
    assert float(off.abs().max()) < 32, "two distinct ids produced the same code"
    assert abs(float(off.sum()) / (4096 * 4095)) < 0.5


def test_direct_mode_allocates_no_table():
    module = sidecar("direct")
    assert module.codes is None
    assert list(module.named_buffers()) == []
    report = module.parameter_report()
    assert report["frozen_code_entries"] == 0
    # Same trainable footprint as the stored-basis arm it is replacing, so the
    # comparison is about representation rather than capacity.
    assert report["trainable_parameters"] == sidecar("fixed").parameter_report()[
        "trainable_parameters"]


def test_direct_mode_is_addressed_and_respects_wrong_context():
    module = sidecar("direct")
    module.decoder[-1].weight.data.normal_(0, 0.5)
    rows = hasher().row_indices(ids(length=16))
    assert not torch.equal(module(rows), module(wrong_context_rows(rows)))


def test_direct_mode_starts_at_exactly_zero():
    module = sidecar("direct")
    rows = hasher().row_indices(ids())
    assert torch.equal(module(rows), torch.zeros(2, 9, len(STRUCTURAL)))


def test_direct_features_survive_a_round_trip_to_another_device():
    """CPU and GPU must agree, or a benchmark would compare two different functions."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    rows = torch.arange(256, dtype=torch.int64).reshape(1, 128, 2)
    host = signed_hash_features(rows, 32, 7)
    device = signed_hash_features(rows.cuda(), 32, 7).cpu()
    assert torch.equal(host, device)


def test_a_feature_width_beyond_one_hash_word_is_refused():
    with pytest.raises(ValueError, match="between 1 and 32"):
        signed_hash_features(torch.zeros(1, 2, 2, dtype=torch.int64), 64, seed=1)


# --- factorization --------------------------------------------------------------
#
# Two controls said "structural" was hiding two mechanisms: the unaddressed baseline kept
# 83% of the whitespace gain and 12% of newline, and cross-backbone transfer kept 16% of
# whitespace and 89% of newline. Splitting the module along that line is only a test of
# that claim if the two halves are genuinely separate -- an addressed branch that still
# writes whitespace, or a bias that quietly depends on the hash, would produce the same
# numbers and mean nothing.

STRUCTURAL_IDS = torch.tensor(STRUCTURAL, dtype=torch.long)
WHITESPACE_IDS = torch.tensor([STRUCTURAL[1], STRUCTURAL[3]], dtype=torch.long)


def factorized(mode="direct", trained=True):
    addressed = sidecar(mode)
    if trained:
        addressed.decoder[-1].weight.data.normal_(0, 0.5)
        addressed.decoder[-1].bias.data.normal_(0, 0.5)
    module = FactorizedSidecar(addressed, STRUCTURAL_IDS, WHITESPACE_IDS)
    module.white.bias.data.normal_(0, 0.5)
    return module


def slots():
    positions = {value: index for index, value in enumerate(STRUCTURAL)}
    return torch.tensor([positions[int(value)] for value in WHITESPACE_IDS])


def others():
    keep = [index for index, value in enumerate(STRUCTURAL)
            if value not in set(WHITESPACE_IDS.tolist())]
    return torch.tensor(keep)


def test_the_addressed_branch_writes_nothing_to_whitespace():
    module = factorized()
    rows = hasher().row_indices(ids(length=16))
    out = module(rows, 0.0)
    assert torch.equal(out[..., slots()], torch.zeros(2, 16, len(WHITESPACE_IDS)))
    assert torch.any(out[..., others()] != 0)


def test_the_whitespace_branch_writes_nothing_outside_whitespace():
    module = factorized()
    rows = hasher().row_indices(ids(length=16))
    only_addressed = module(rows, 0.0)
    with_bias = module(rows, 1.0)
    assert torch.equal(with_bias[..., others()], only_addressed[..., others()])
    assert torch.any(with_bias[..., slots()] != 0)


def test_the_whitespace_branch_ignores_the_addresses():
    """Context-free means context-free: the same bias wherever the hash points."""
    module = factorized()
    rows = hasher().row_indices(ids(length=16))
    wrong = wrong_context_rows(rows)
    assert torch.equal(module(rows, 1.0)[..., slots()],
                       module(wrong, 1.0)[..., slots()])
    assert not torch.equal(module(rows, 1.0)[..., others()],
                           module(wrong, 1.0)[..., others()])


def test_each_branch_can_be_disabled_independently():
    module = factorized()
    rows = hasher().row_indices(ids(length=16))
    assert torch.equal(module(rows, 0.0)[..., slots()],
                       torch.zeros(2, 16, len(WHITESPACE_IDS)))
    module.addressed.decoder[-1].weight.data.zero_()
    module.addressed.decoder[-1].bias.data.zero_()
    out = module(rows, 1.0)
    assert torch.equal(out[..., others()], torch.zeros(2, 16, len(others())))


def test_the_whitespace_strength_scales_only_that_branch():
    module = factorized()
    rows = hasher().row_indices(ids(length=16))
    half = module(rows, 0.5)
    full = module(rows, 1.0)
    assert torch.allclose(half[..., slots()], 0.5 * full[..., slots()])
    assert torch.equal(half[..., others()], full[..., others()])


def test_fitting_the_bias_leaves_the_addressed_decoder_alone():
    module = factorized()
    rows = hasher().row_indices(ids(length=16))
    before = {name: value.clone()
              for name, value in module.addressed.state_dict().items()}
    module.addressed.requires_grad_(False)
    optimizer = torch.optim.AdamW(module.white.parameters(), lr=0.1, weight_decay=0.5)
    module(rows, 1.0).sum().backward()
    optimizer.step()
    after = module.addressed.state_dict()
    for name, value in before.items():
        assert torch.equal(after[name], value), name
    assert module.white.bias.grad is not None


def test_the_whitespace_branch_is_one_parameter_per_token():
    module = factorized()
    report = module.parameter_report()
    assert report["whitespace_parameters"] == len(WHITESPACE_IDS)
    assert report["whitespace_tokens"] == len(WHITESPACE_IDS)
    assert report["total_parameters"] == (report["addressed_parameters"]
                                          + report["whitespace_parameters"])


def test_a_whitespace_id_outside_the_structural_set_is_refused():
    with pytest.raises(ValueError, match="not in the structural set"):
        FactorizedSidecar(sidecar("direct"), STRUCTURAL_IDS,
                          torch.tensor([999], dtype=torch.long))


def test_a_fresh_whitespace_bias_contributes_nothing():
    module = FactorizedSidecar(sidecar("direct"), STRUCTURAL_IDS, WHITESPACE_IDS)
    assert torch.equal(module.white(), torch.zeros(len(WHITESPACE_IDS)))
    with pytest.raises(ValueError, match="at least one token"):
        WhitespaceBias(0)
