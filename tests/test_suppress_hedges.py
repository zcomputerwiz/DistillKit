"""The hedge-suppressed teacher target: what it removes, what it keeps, what it sums to."""
import sys

import numpy as np

sys.path.insert(0, "scratch/dense_gr")
from teacher_kl import suppress_teacher_tokens  # noqa: E402

WAIT, OTHER = 7, 3


def rows():
    # Four positions, top-3 each. Position p's row predicts ids[p + 1].
    ids = np.array([1, 2, WAIT, 4, 5])
    topk = np.array([[WAIT, 2, 9], [WAIT, 8, 9], [WAIT, 4, 9], [OTHER, WAIT, 5], [5, 6, 9]])
    probs = np.array([[0.5, 0.3, 0.1], [0.2, 0.5, 0.2], [0.3, 0.6, 0.05],
                      [0.1, 0.4, 0.4], [0.9, 0.05, 0.01]])
    return ids, topk, np.log(probs).astype(np.float32)


def test_removes_hedges_only_in_the_answer_and_only_where_the_text_does_not_hedge():
    ids, topk, lp = rows()
    out, removed = suppress_teacher_tokens(ids, topk, lp, np.array([WAIT]), start=2)
    # Row 0 predicts token 1 (prompt side of start=2): untouched.
    assert np.array_equal(out[0], lp[0])
    # Row 1 predicts ids[2] == WAIT: the text hedges there, so the teacher stays whole.
    assert np.array_equal(out[1], lp[1])
    # Rows 2 and 3 predict non-hedges: the WAIT entry is gone.
    assert out[2, 0] == -1e4 and out[3, 1] == -1e4
    assert np.isclose(removed, 0.3 + 0.4)


def test_the_rest_keeps_its_proportions_and_the_tail_scales_with_it():
    ids, topk, lp = rows()
    out, _ = suppress_teacher_tokens(ids, topk, lp, np.array([WAIT]), start=0)
    before, after = np.exp(lp[2]), np.exp(out[2].astype(np.float64))
    kept = [1, 2]
    assert np.allclose(after[kept], before[kept] / (1 - before[0]), rtol=1e-5)
    tail_before, tail_after = 1 - before.sum(), 1 - after[kept].sum()
    assert np.isclose(tail_after, tail_before / (1 - before[0]), rtol=1e-4)
    assert after[0] == 0.0


def test_nothing_to_remove_returns_the_input():
    ids, topk, lp = rows()
    out, removed = suppress_teacher_tokens(ids, topk, lp, np.array([42]), start=0)
    assert out is lp and removed == 0.0
