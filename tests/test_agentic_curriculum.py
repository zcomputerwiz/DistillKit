# Assisted-by: Codex
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch/dense_gr"))
from agentic_curriculum import DOMAINS, KINDS, CONTRAST_KINDS, make_trajectory, make_contrast, validate
from teacher_kl import CachedTeacher
from distillkit.offline_cache import OfflineCacheWriter
from agentic_live_eval import Environment


def test_references_execute_and_discovery_precedes_mutation():
    for domain in DOMAINS:
        for kind in KINDS:
            row = make_trajectory(domain, kind, 17)
            validate(row)
            calls = [c["function"]["name"] for m in row["messages"] for c in m.get("tool_calls", [])]
            if kind in ("discover", "disambiguate", "ask_choice", "empty_search", "read_after_search"):
                assert calls[0].endswith("_search")
            if kind == "recover":
                assert calls == [domain + "_set_state", domain + "_refresh", domain + "_set_state"]
            if kind in ("no_call", "already_done", "empty_search"):
                assert not any(c.endswith("_set_state") for c in calls)


def test_announcement_without_call_and_bad_schema_rejected():
    row = make_trajectory("catalog", "announce", 1)
    bad = copy.deepcopy(row)
    next(m for m in bad["messages"] if m.get("tool_calls"))["tool_calls"] = []
    with pytest.raises(AssertionError):
        validate(bad)
    bad = copy.deepcopy(row)
    next(m for m in bad["messages"] if m.get("tool_calls"))["tool_calls"][0]["function"]["arguments"] = {"invented": 1}
    with pytest.raises(AssertionError):
        validate(bad)


def test_ambiguity_uses_available_section_before_asking():
    resolved = make_trajectory("archive", "disambiguate", 2)
    unresolved = make_trajectory("archive", "ask_choice", 2)
    assert not any("Which section" in m.get("content", "") for m in resolved["messages"])
    question = next(i for i, m in enumerate(unresolved["messages"]) if "Which section" in m.get("content", ""))
    assert unresolved["messages"][question - 1]["role"] == "tool"
    assert unresolved["messages"][question + 1]["role"] == "user"


def test_hard_label_cache_requires_ce_and_role_mask(tmp_path):
    cache = tmp_path / "cache"
    with OfflineCacheWriter(cache, tokenizer_hash="0" * 64, anchor_layers=[], hidden_size=4,
                            vocab_size=16, top_k=2, sequence_length=8, shard_tokens=8,
                            metadata={"target_kind": "hard_labels_only"}) as writer:
        ids = np.array([5, 1, 2, 4, 3, 6])
        writer.append("train", ids, np.tile([7, 8], (6, 1)), np.full((6, 2), -3.))
    options = dict(device="cpu", answer_marker=[1, 2], turn_close=3)
    with pytest.raises(ValueError, match="requires ce_only"):
        CachedTeacher(cache, **options)
    with pytest.raises(ValueError, match="requires assistant_only"):
        CachedTeacher(cache, ce_only=[cache], **options)
    teacher = CachedTeacher(cache, ce_only=[cache], assistant_only=[cache], **options)
    record = teacher.read_batch(["train"], 6)
    assert record["ce_only"]
    assert record["weight"].tolist() == [[0., 0., 1., 1., 0., 0.]]


def test_live_reference_trajectories_finish_and_announcements_do_not_execute():
    for kind in KINDS:
        row = make_trajectory("catalog", kind, 3)
        env = Environment(row)
        for m in row["messages"]:
            if m["role"] != "assistant":
                continue
            text = m["content"] + "".join("<tool_call>" + json.dumps({
                "name": c["function"]["name"], "arguments": c["function"]["arguments"]}) + "</tool_call>"
                for c in m.get("tool_calls", []))
            env.respond(text)
        assert env.success and env.done, (kind, env.errors)
    env = Environment(make_trajectory("catalog", "known_id", 3))
    env.respond("I updated the record.")
    assert env.done and not env.success and env.mutations == 0


def test_live_extra_read_allowed_but_unresolved_choice_cannot_mutate():
    row = make_trajectory("catalog", "known_id", 3)
    env = Environment(row)
    from tool_behavior_eval import call
    env.execute(call("catalog_read", {"record_id": row["environment"]["id"]}))
    env.execute(call("catalog_set_state", {"record_id": row["environment"]["id"],
                                         "state": row["environment"]["desired_state"]}))
    env.respond("Updated successfully.")
    assert env.success
    row = make_trajectory("catalog", "ask_choice", 3)
    env = Environment(row)
    result = env.execute(call("catalog_set_state", {"record_id": row["environment"]["id"],
                                                   "state": row["environment"]["desired_state"]}))
    assert result["error"] == "UNREQUESTED_MUTATION"
    assert env.mutations == 0


def test_v2_pairs_condition_write_on_observation_and_all_schema_variants_execute():
    for n in range(4):
        done = make_contrast("catalog", "already_done", n)
        needed = make_contrast("catalog", "check_needed", n)
        assert done["tools"] == needed["tools"]
        first_result = next(i for i, m in enumerate(done["messages"]) if m["role"] == "tool")
        assert done["messages"][:first_result] == needed["messages"][:first_result]
        assert done["environment"]["initial_state"] != needed["environment"]["initial_state"]
        for kind in CONTRAST_KINDS:
            row = make_contrast("catalog", kind, n)
            validate(row)
            env = Environment(row)
            for m in row["messages"]:
                if m["role"] != "assistant":
                    continue
                text = m["content"] + "".join("<tool_call>" + json.dumps({
                    "name": c["function"]["name"], "arguments": c["function"]["arguments"]}) + "</tool_call>"
                    for c in m.get("tool_calls", []))
                env.respond(text)
            assert env.success, (n, kind, env.errors)
            if kind in ("known_read", "already_done", "check_needed"):
                assert not env.searched


def test_v2_read_only_target_does_not_authorize_write():
    row = make_contrast("catalog", "known_read", 2)
    env = Environment(row)
    from tool_behavior_eval import call
    setter = next(name for name, op in row["environment"]["operations"].items() if op == "set_state")
    result = env.execute(call(setter, {row["environment"]["id_argument"]: row["environment"]["id"],
                                      "state": row["environment"]["desired_state"]}))
    assert result["error"] == "UNREQUESTED_MUTATION" and env.mutations == 0


def test_smaller_recipe_changes_only_requested_replay_weights_and_rate(tmp_path):
    from agentic_arm import recipe
    original, old = recipe(tmp_path / "data", tmp_path / "old")
    argv, new = recipe(tmp_path / "data", tmp_path / "new", steps=20,
                       code_multiplier=3, rate_scale=.275)
    code = {"teacher-cache-frontier-code-raw", "teacher-cache-r8-code-short-w8",
            "teacher-cache-expand-code-w8", "teacher-cache-teacher-code"}
    for path, repeats in old["repeat"].items():
        assert new["repeat"][path] == repeats * (3 if Path(path).name in code else 1)
    assert argv[argv.index("--max-steps") + 1] == "20"
    assert argv[argv.index("--lr-depth-ramp") + 1:argv.index("--lr-depth-ramp") + 3] == ["0.275", "0.275"]
    assert new["ce"] == old["ce"] and new["assistant"] == old["assistant"]


def test_conversational_replay_mask_preserves_raw_sources_and_objectives(tmp_path):
    from agentic_arm import recipe
    _, old = recipe(tmp_path / 'data', tmp_path / 'old')
    _, new = recipe(tmp_path / 'data', tmp_path / 'new', mask_conversational=True)
    raw = {'teacher-cache-frontier-code-raw', 'teacher-cache-general-pilot-w8'}
    assert {Path(p).name for p in new['assistant']} == {Path(p).name for p in new['paths']} - raw
    for key in ('ce', 'kl', 'ul', 'repeat'):
        assert new[key] == old[key]


def test_preservation_recipe_freezes_router_and_uses_the_same_exclusions_for_measurement(tmp_path):
    from agentic_arm import recipe, checkpoint_name
    exclusion = tmp_path/'confirmed-exclusions.json'
    exclusion.write_text('[]', encoding='ascii')
    _, legacy = recipe(tmp_path/'data', tmp_path/'legacy', control=True, mask_conversational=True)
    argv, options = recipe(tmp_path/'data', tmp_path/'next', control=True, mask_conversational=True,
                           freeze_router=True, exclude_documents=exclusion)
    assert '--freeze-router' in argv
    assert argv[argv.index('--exclude-documents')+1] == options['exclude_documents'] == str(exclusion.resolve())
    assert checkpoint_name(True).endswith('-csa2-frozen')
    assert checkpoint_name(False).endswith('-csa2')
    for key in ('paths', 'ce', 'kl', 'ul', 'repeat', 'assistant'):
        assert options[key] == legacy[key]
