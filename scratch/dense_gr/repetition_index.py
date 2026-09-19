"""The repetition index: where the current suffix occurred before, computed exactly.

This is the first standard part, and it is deliberately the most boring one available.
`docs/standard_parts.md` selects it on four criteria -- universal, exactly computable,
load-bearing under substitution, portable -- and it is the only candidate that scores well
on all four with no learning anywhere in it. It is what induction heads compute, and a
dictionary computes it in O(1) per token.

For each position ``t`` the module reports what a suffix match of the last ``order``
tokens finds:

* ``matched``    -- whether this suffix has been seen before in the sequence
* ``distance``   -- how far back the most recent previous occurrence was
* ``length``     -- how long the agreeing suffix is, capped at ``max_length``
* ``candidate``  -- the token that followed that occurrence, which is the induction
  head's answer

The features are a fixed-width code by construction, which satisfies the portability
constraint: nothing here has a shape that depends on ``d_model``, so the same module
attaches to a backbone of any width through that backbone's own projection. ``candidate``
is a token id and therefore lives in the tokenizer ABI, which the document already
declares as a fixed boundary.

**The causality rule.** Position ``t`` may only consult positions strictly before ``t``.
A module that peeks at its own continuation hands the backbone the answer and every
downstream measurement becomes meaningless -- so the table is updated only after a
position has been scored, and `_self_check` tests exactly that.
"""

from __future__ import annotations

import numpy as np

FEATURES = 4  # matched, log distance, length, candidate-agrees-with-previous

#: Distance is normalized by a constant, not by the window length. Dividing by log1p(n)
#: made the same context yield different features at different window sizes, which broke
#: prefix-invariance and would break incremental decoding, where n is not known ahead.
SCALE = float(np.log1p(4096))


def repetition_index(tokens, order=3, max_length=8):
    """Exact suffix matching over one sequence of ids.

    Returns ``(features, candidate)``: ``features`` is ``(length, FEATURES)`` float32 and
    ``candidate`` is ``(length,)`` int64, holding the continuation token of the most
    recent previous occurrence of the current ``order``-gram, or ``-1`` where there is
    none.

    The hash table maps an ``order``-gram to the last position at which it *started*, so
    the continuation of that occurrence is the token at ``start + order``. Only n-grams
    whose continuation has already been seen are inserted, which is what keeps the whole
    thing causal.
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    n = tokens.shape[0]
    features = np.zeros((n, FEATURES), dtype=np.float32)
    candidate = np.full(n, -1, dtype=np.int64)

    last_start = {}
    previous_candidate = -1
    for t in range(n):
        # The suffix ending at t - 1 is what predicts t, so the gram we look up starts at
        # t - order and ends at t - 1. It needs `order` tokens to exist behind it.
        if t >= order:
            gram = tokens[t - order:t].tobytes()
            hit = last_start.get(gram)
            if hit is not None:
                following = hit + order
                if following < t:
                    found = int(tokens[following])
                    candidate[t] = found
                    distance = t - following
                    length = _agreement(tokens, t, following, max_length)
                    features[t, 0] = 1.0
                    features[t, 1] = np.log1p(distance) / SCALE
                    features[t, 2] = length / max_length
                    features[t, 3] = 1.0 if found == previous_candidate else 0.0
            previous_candidate = int(candidate[t])
            # Insert only after scoring, so position t never sees itself.
            last_start[gram] = t - order
    return features, candidate


def _agreement(tokens, t, following, max_length):
    """How far back the two suffixes agree, capped -- a longer match is a stronger one."""
    length = 0
    while (length < max_length and t - 1 - length >= 0 and following - 1 - length >= 0
           and tokens[t - 1 - length] == tokens[following - 1 - length]):
        length += 1
    return length


def batch_repetition_index(batch, order=3, max_length=8):
    """`repetition_index` over a ``(batch, length)`` array of ids."""
    batch = np.asarray(batch, dtype=np.int64)
    features = np.zeros(batch.shape + (FEATURES,), dtype=np.float32)
    candidate = np.full(batch.shape, -1, dtype=np.int64)
    for row in range(batch.shape[0]):
        features[row], candidate[row] = repetition_index(batch[row], order, max_length)
    return features, candidate


def _self_check():
    """Causality first, then that it actually finds the repeat it is supposed to find."""
    rng = np.random.default_rng(0)

    # Causality: truncating the sequence must not change any feature that survives. If a
    # position could see its own future, the prefix and the full sequence would disagree.
    sequence = rng.integers(0, 50, size=400)
    full, full_candidate = repetition_index(sequence)
    for cut in (37, 111, 250):
        part, part_candidate = repetition_index(sequence[:cut])
        assert np.array_equal(full[:cut], part), "position saw beyond its prefix at %d" % cut
        assert np.array_equal(full_candidate[:cut], part_candidate)

    # Function: a random block repeated once. Inside the repeat the candidate must be the
    # true next token -- that is the whole claim of the part.
    half = 64
    block = rng.integers(0, 1000, size=half)
    repeated = np.concatenate([block, block])
    _, found = repetition_index(repeated, order=3)
    # The first `order` positions of the repeat cannot match yet; after that it is exact.
    inside = slice(half + 3, 2 * half)
    truth = repeated[inside]
    assert np.array_equal(found[inside], truth), (found[inside][:10], truth[:10])

    # And on non-repeating input it must mostly find nothing, or it is matching noise.
    unique = rng.integers(0, 100_000, size=400)
    _, nothing = repetition_index(unique, order=3)
    assert (nothing >= 0).mean() < 0.02, (nothing >= 0).mean()

    print("repetition_index self-check: ok (%d/%d exact inside the repeat, %.3f%% "
          "false matches on unique input)"
          % ((found[inside] == truth).sum(), truth.shape[0],
             100 * (nothing >= 0).mean()))


if __name__ == "__main__":
    _self_check()
