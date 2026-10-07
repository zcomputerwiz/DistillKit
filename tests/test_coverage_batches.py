# Assisted-by: Codex
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch/dense_gr"))
from training_state import PlannedBatches, coverage_order
from training_state import OrderedBatches
import hashlib
import json


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


def test_ordered_batches_are_finite_resume_exactly_and_bind_cache():
    teacher, groups = fixture()
    teacher.group_weight = lambda docs, width: len(docs)*(width-1)
    manifest = dict(groups=[dict(documents=groups[i][0],width=10) for i in [9,2,7]],
        cache_sha256=hashlib.sha256(json.dumps([{}],sort_keys=True).encode()).hexdigest())
    batches = OrderedBatches(teacher,groups,manifest)
    assert batches.take(1000) == groups[9]
    state = batches.state_dict()
    restored = OrderedBatches(teacher,groups,manifest)
    restored.load_state_dict(state)
    assert restored.take(1000) == groups[2]
    assert restored.take(1000) == groups[7]
    assert restored.take(float('inf')) is None
    assert restored.next_targets() == float('inf')
    bad = dict(manifest,cache_sha256='changed')
    with pytest.raises(ValueError,match='cache manifests'):
        OrderedBatches(teacher,groups,bad)
    bad = dict(manifest,groups=[dict(documents=['missing'],width=10)])
    with pytest.raises(ValueError,match='canonical'):
        OrderedBatches(teacher,groups,bad)
