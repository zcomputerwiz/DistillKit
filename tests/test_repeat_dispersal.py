"""Repeated documents: each copy is its own visit, never a second row of the same microbatch."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch" / "dense_gr"))

from teacher_kl import CachedTeacher  # noqa: E402


def plan(ids, size, cap=lambda doc_id: 100):
    stub = SimpleNamespace(ids=ids, cap=cap, pad_blocks=False, real_width={}, min_answer_tokens=0,
                           kl_only_ids=set(), unlikelihood_ids=set(), ce_only_ids=set())
    return CachedTeacher._groups(stub, size)


def test_copies_of_a_document_never_share_a_group():
    ids = ["d%d" % i for i in range(10) for _ in range(3)] + ["once%d" % i for i in range(5)]
    groups = plan(ids, 4)
    assert sorted(d for group, _ in groups for d in group) == sorted(ids)
    assert all(len(set(group)) == len(group) for group, _ in groups)


def test_sparse_buckets_and_unequal_repeats_close_batches_short():
    # Codex recheck r5: a bucket with fewer documents than rows put copies side by side.
    assert plan(["a"] * 3, 4) == [(["a"], 100), (["a"], 100), (["a"], 100)]
    groups = plan(["a"] * 3 + ["b"] + ["c"] * 2, 2)
    assert sorted(d for g, _ in groups for d in g) == ["a", "a", "a", "b", "c", "c"]
    assert all(len(set(g)) == len(g) for g, _ in groups)
    many = ["d%d" % (i % 7) for i in range(7 * 5)] + ["e%d" % i for i in range(3) for _ in range(2)]
    assert all(len(set(g)) == len(g) for g, _ in plan(many, 6))


def test_prefix_exposure_replays_the_plan_budget():
    from training_state import PlannedBatches, prefix_exposure

    ids = ["a%d" % i for i in range(6) for _ in range(2)] + ["b%d" % i for i in range(6)]
    groups = plan(ids, 2)
    owner = {d: ("caps/" + d[0], None) for d in ids}
    teacher = SimpleNamespace(cache=SimpleNamespace(_owner=owner), real_width={},
                              group_weight=lambda group, width: len(group) * (width - 1))
    seen = prefix_exposure(teacher, groups, seed=3, budget=10 * 99 + 50)
    # The planner takes whole two-row batches: five fit in 1,040 targets.
    assert sum(v for v, _, _ in seen.values()) == 10 and sum(t for _, _, t in seen.values()) == 990
    assert all(distinct <= visits for visits, distinct, _ in seen.values())
    real = PlannedBatches(teacher, groups, 3)
    taken = []
    while (batch := real.next_targets()) <= 1040 - 99 * len(taken):
        taken += groups[real.order[real.cursor]][0]
        real.cursor += 1
    assert {k: v[0] for k, v in seen.items()} == {k: sum(d[0] == k for d in taken) for k in seen}


def test_group_shapes_do_not_depend_on_repeats_order():
    # The same members per bucket chunk into the same row counts: warmed shapes hold.
    ids = ["d%d" % i for i in range(7) for _ in range(2)]
    widths = lambda doc_id: 100 if int(doc_id[1:]) % 2 else 200
    assert sorted((len(g), w) for g, w in plan(ids, 3, widths)) == [(2, 200), (3, 100), (3, 100),
                                                                    (3, 200), (3, 200)]
