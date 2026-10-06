# Assisted-by: Codex
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch/dense_gr"))
from training_state import PlannedBatches, coverage_order


def fixture():
    owners = {str(i): ("common" if i < 98 else "rare" + str(i), None) for i in range(100)}
    cache = SimpleNamespace(_owner=owners, manifest={})
    teacher = SimpleNamespace(cache=cache, read_batch=lambda docs, width: (docs, width))
    return teacher, [([str(i)], 10) for i in range(100)]


def test_coverage_is_a_permutation_includes_rare_sources_and_is_deterministic():
    teacher, groups = fixture()
    order = coverage_order(teacher, groups, np.random.default_rng(25), 10)
    assert sorted(order) == list(range(100))
    assert {teacher.cache._owner[groups[i][0][0]][0] for i in order[:10]} == {"common", "rare98", "rare99"}
    assert order == coverage_order(teacher, groups, np.random.default_rng(25), 10)
    with pytest.raises(ValueError, match="per source"):
        coverage_order(teacher, groups, np.random.default_rng(25), 2)


def test_resume_preserves_prefix_and_epoch_transition_and_rejects_changed_balance():
    teacher, groups = fixture()
    original = PlannedBatches(teacher, groups, 25, 10)
    for _ in range(7):
        original.take(1000)
    state = original.state_dict()
    resumed = PlannedBatches(teacher, groups, 25, 10)
    resumed.load_state_dict(state)
    assert [original.take(1000) for _ in range(105)] == [resumed.take(1000) for _ in range(105)]
    with pytest.raises(ValueError, match="sample plan"):
        PlannedBatches(teacher, groups, 25, 11).load_state_dict(state)
    ordinary = PlannedBatches(teacher, groups, 25)
    expected = np.random.default_rng(25).permutation(100)
    assert [int(ordinary.take(1000)[0][0]) for _ in range(100)] == list(expected)
