"""The row-novelty stratification, which decides whether the sidecar generalises.

The C arms showed the layer-1 sidecar beating its shuffled control on assistant NLL.
That result splits by how many of a position's 16 table rows the training run had
touched, and the split only means anything if two things are right: ``k`` counts rows
against the training set correctly, and it is attached to the position that produced
each token's loss rather than the position of the token itself. An off-by-one there
changes no total and scrambles every stratum, so it is pinned here.
"""

import numpy as np
import pytest
import torch

from distillkit.ngram_hash import NGramHasher

row_novelty = pytest.importorskip("scratch.row_novelty")


@pytest.fixture(scope="module")
def hasher():
    return NGramHasher()


def _rows(hasher, ids):
    return hasher.row_indices(torch.tensor(ids, dtype=torch.long).unsqueeze(0))[0].numpy()


def test_a_position_whose_rows_were_all_trained_scores_sixteen(hasher):
    ids = [17, 4242, 9, 300, 7]
    seen = np.unique(_rows(hasher, ids).ravel())
    assert (row_novelty.novelty_per_position(ids, seen, hasher) == 16).all()


def test_a_position_whose_rows_are_all_new_scores_zero(hasher):
    ids = [17, 4242, 9, 300, 7]
    # Rows from a different token stream. Distinct contexts can still collide in a
    # hashed table, so subtract this stream's own rows rather than assume they differ.
    other = np.unique(_rows(hasher, [11, 12, 13, 14, 15]).ravel())
    seen = np.setdiff1d(other, np.unique(_rows(hasher, ids).ravel()))
    assert (row_novelty.novelty_per_position(ids, seen, hasher) == 0).all()


def test_partial_exposure_counts_exactly_the_trained_heads(hasher):
    ids = [17, 4242, 9, 300, 7]
    rows = _rows(hasher, ids)
    # Train five of position 2's sixteen heads and nothing else.
    seen = np.unique(rows[2, :5])
    novelty = row_novelty.novelty_per_position(ids, seen, hasher)
    assert novelty[2] >= 5
    assert novelty[2] == int(np.isin(rows[2], seen).sum()), "miscounted its own heads"


def test_membership_survives_rows_outside_the_trained_range(hasher):
    """searchsorted clamps at the end of the array; an unclamped index would read past
    it, and a row above every trained row would be reported as a hit."""
    ids = [17, 4242, 9, 300, 7]
    rows = _rows(hasher, ids)
    seen = np.array([rows.min() - 1], dtype=np.int64)
    assert (row_novelty.novelty_per_position(ids, seen, hasher) == 0).all()


def test_a_token_takes_the_novelty_of_the_position_that_predicted_it(hasher):
    """The loss on target ``i`` comes from the logits at position ``i - 1``."""
    ids = [17, 4242, 9, 300, 7, 88, 91]
    rows = _rows(hasher, ids)
    seen = np.unique(rows[3])          # position 3 alone is fully trained
    targets = np.array([3, 4, 5])
    novelty = row_novelty.novelty_for_targets(ids, targets, seen, hasher)
    assert novelty[1] == 16, "target 4 is predicted from position 3 and should be seen"
    assert novelty[0] < 16 and novelty[2] < 16, "novelty attached to the wrong position"


def test_strata_partition_the_tokens():
    """Every scored token lands in exactly one stratum, so the strata's token counts
    have to add back to the total the aggregate result was computed over."""
    packed = {"doc": np.array([0, 0, 1, 1, 1]), "k": np.array([0, 16, 0, 8, 16]),
              "enabled": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
              "bypassed": np.array([1.5, 1.0, 3.5, 4.5, 4.0])}
    total = 0
    for k in (0, 8, 16):
        _, _, tokens = row_novelty.by_document(packed, packed["k"] == k, 2)
        total += tokens.sum()
    assert total == len(packed["k"])


def test_per_document_sums_are_what_the_bootstrap_resamples():
    """Resampling tokens would ignore the correlation within a document and give
    intervals far too tight; the unit has to be the document."""
    packed = {"doc": np.array([0, 0, 1]), "k": np.zeros(3, dtype=np.int8),
              "enabled": np.array([1.0, 2.0, 3.0]), "bypassed": np.array([0.5, 1.0, 2.0])}
    enabled, bypassed, tokens = row_novelty.by_document(packed, packed["k"] == 0, 2)
    assert list(enabled) == [3.0, 3.0]
    assert list(bypassed) == [1.5, 2.0]
    assert list(tokens) == [2.0, 1.0]


def test_the_paired_gap_reduces_to_the_difference_of_the_two_arms():
    left, right = np.array([2.0, 4.0]), np.array([1.0, 1.0])
    tokens = np.array([1.0, 1.0])
    estimate, low, high = row_novelty.paired_gap(left, right, tokens, tokens, draws=200)
    assert estimate == pytest.approx(3.0 - 1.0)
    assert low <= estimate <= high


def test_an_empty_stratum_in_a_draw_does_not_poison_the_interval():
    """Most documents contribute nothing to a thin stratum, so some bootstrap draws
    have a zero denominator. Those must drop out rather than return inf or nan."""
    left = np.array([1.0, 0.0, 0.0, 0.0])
    tokens = np.array([1.0, 0.0, 0.0, 0.0])
    estimate, low, high = row_novelty.bootstrap(left, tokens, draws=500)
    assert np.isfinite([estimate, low, high]).all()
