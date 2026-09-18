"""Vectorized repetition index, validated against the reference implementation.

`repetition_index.py` is the readable definition: a dictionary, one position at a time,
obviously causal. It costs 0.118 s per training step at batch 64 x 1024, which is 21% of a
0.56 s step -- enough to matter and enough to be worth removing rather than hiding behind a
prefetch thread.

The loop is only apparently sequential. "The most recent earlier position with the same
n-gram" is a group-by, not a scan: sort positions by n-gram id, and within each group every
position's answer is simply the element before it. That is one `argsort` and a shift.

Two details make the encoding exact rather than a hash. An `order`-gram over a vocabulary
of `V` is a base-`V` integer, so it fits an `int64` whenever `V**order < 2**63` -- at the
32,768 cut and order 3 that is 2**45, with room to spare. And the sort is stable, so equal
n-grams keep their positional order and "the element before it in the group" really is the
most recent earlier occurrence. No collisions, no approximation: this computes the same
function as the reference, which `_self_check` asserts on random and adversarial input.
"""

from __future__ import annotations

import numpy as np

FEATURES = 4


def batch_repetition_index(batch, order=3, max_length=8, vocab=None):
    """`(features, candidate)` for a ``(batch, length)`` array, same result as the loop."""
    batch = np.asarray(batch, dtype=np.int64)
    rows, n = batch.shape
    if vocab is None:
        vocab = int(batch.max()) + 1 if batch.size else 1
    if vocab ** order >= 2 ** 62:
        raise SystemExit("vocabulary %d at order %d overflows an exact gram id"
                         % (vocab, order))

    features = np.zeros((rows, n, FEATURES), dtype=np.float32)
    candidate = np.full((rows, n), -1, dtype=np.int64)
    if n <= order:
        return features, candidate

    # gram[t] is the order-gram ending at t-1, the suffix that predicts t.
    windows = np.lib.stride_tricks.sliding_window_view(batch, order, axis=1)[:, :n - order]
    weights = vocab ** np.arange(order - 1, -1, -1, dtype=np.int64)
    grams = (windows * weights).sum(axis=2)

    positions = np.arange(order, n, dtype=np.int64)
    for row in range(rows):
        previous = _previous_occurrence(grams[row], positions)
        found = previous >= 0
        if not found.any():
            continue
        here = positions[found]
        there = previous[found]

        candidate[row, here] = batch[row, there]
        features[row, here, 0] = 1.0
        features[row, here, 1] = np.log1p(here - there) / np.log1p(n)
        features[row, here, 2] = _agreement(batch[row], here, there, max_length) / max_length

    # Feature 3: the candidate agrees with the previous position's candidate. The reference
    # seeds this with -1, and candidate is -1 wherever nothing matched, so a plain shift
    # reproduces it.
    previous_candidate = np.concatenate(
        [np.full((rows, 1), -1, dtype=np.int64), candidate[:, :-1]], axis=1)
    agrees = (candidate == previous_candidate) & (candidate >= 0)
    features[:, :, 3] = agrees.astype(np.float32)
    return features, candidate


def _previous_occurrence(grams, positions):
    """For each position, the most recent earlier position holding the same gram."""
    sorter = np.argsort(grams, kind="stable")
    ordered = grams[sorter]
    result = np.full(positions.shape[0], -1, dtype=np.int64)
    same = np.zeros(positions.shape[0], dtype=bool)
    same[1:] = ordered[1:] == ordered[:-1]
    shifted = np.full(positions.shape[0], -1, dtype=np.int64)
    shifted[1:] = positions[sorter][:-1]
    result[sorter] = np.where(same, shifted, -1)
    return result


def _agreement(tokens, here, there, max_length):
    """How far back the two suffixes agree, capped at `max_length`."""
    length = np.zeros(here.shape[0], dtype=np.int64)
    alive = np.ones(here.shape[0], dtype=bool)
    for step in range(max_length):
        left = here - 1 - step
        right = there - 1 - step
        ok = alive & (left >= 0) & (right >= 0)
        ok[ok] = tokens[left[ok]] == tokens[right[ok]]
        length += ok
        alive = ok
        if not alive.any():
            break
    return length


def _self_check():
    """Must agree with the reference loop exactly, including where nothing matches."""
    import repetition_index as reference

    rng = np.random.default_rng(0)
    cases = {
        "random": rng.integers(0, 50, size=(4, 400)),
        "tiny vocabulary": rng.integers(0, 3, size=(4, 400)),      # grams repeat constantly
        "unique": rng.integers(0, 100_000, size=(2, 300)),          # nothing repeats
        "repeated block": None,
        "constant": np.zeros((2, 200), dtype=np.int64),             # every gram identical
    }
    block = rng.integers(0, 1000, size=(3, 128))
    cases["repeated block"] = np.concatenate([block, block], axis=1)

    for name, batch in cases.items():
        batch = np.asarray(batch, dtype=np.int64)
        vocab = int(batch.max()) + 1
        want_f, want_c = reference.batch_repetition_index(batch)
        got_f, got_c = batch_repetition_index(batch, vocab=vocab)
        assert np.array_equal(want_c, got_c), (name, "candidate")
        assert np.allclose(want_f, got_f, atol=1e-6), (
            name, "features", np.abs(want_f - got_f).max())
        print("  %-16s ok  (%d matched of %d)"
              % (name, int((got_c >= 0).sum()), got_c.size))

    print("repetition_fast self-check: agrees with the reference on every case")


if __name__ == "__main__":
    _self_check()
