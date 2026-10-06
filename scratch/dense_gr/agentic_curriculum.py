# Assisted-by: Codex
"""Locally verified agent trajectories; no model calls or external API execution.

The cache is hard-label, assistant-only SFT. Reserved top-k arrays are storage
placeholders, NOT captured predictions; teacher_kl refuses to distil them.
Held-out domains test transfer across schemas, not unseen trajectory templates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from tool_behavior_eval import call, schema

POLICY = ("Complete the user's task using the available tools when needed. Retrieve "
          "missing facts through tools before asking the user. Ask only when the "
          "remaining choice or information cannot be resolved from the available "
          "context and tools. If you announce an action, issue its tool call in "
          "the same response. Report completion only after a successful result.")
DOMAINS = ["catalog", "inventory", "archive", "workspace", "library", "registry",
           "observatory", "workshop"]
KINDS = ["discover", "known_id", "disambiguate", "ask_choice", "empty_search",
         "recover", "no_call", "read_after_search", "announce", "already_done"]
CONTRAST_KINDS = KINDS + ["known_read", "check_needed"]


def make_contrast(domain, kind, number):
    """Version 2: paired conditional updates, known-ID reads, and varied schemas.

    The v1 generator and its frozen evaluations remain unchanged. Extra reads
    are valid, but a name search is not a prerequisite when the ID is known.
    """
    base_kind = {"known_read": "read_after_search", "check_needed": "already_done"}.get(kind, kind)
    row = make_trajectory(domain, base_kind, 10000 + number)
    e = row["environment"]
    at = next(i for i, m in enumerate(row["messages"]) if m["role"] == "user")
    prefix = row["messages"][:at]
    user = lambda text: {"role": "user", "content": text}
    assistant = lambda text: {"role": "assistant", "content": text}
    def exchange(operation, arguments, result, ident):
        c = call(domain + "_" + operation, arguments)
        c["id"] = ident
        return [{"role": "assistant", "content": "", "tool_calls": [c]},
                {"role": "tool", "name": domain + "_" + operation, "tool_call_id": ident,
                 "content": json.dumps(result, sort_keys=True)}]

    if kind == "known_read":
        text = [f"What is the current state of record ID {e['id']}?",
                f"Read {e['id']} and report its state. The ID is already known.",
                f"Look up record {e['id']} by ID and tell me its state.",
                f"Please check the state for ID {e['id']}; do not modify it."][number % 4]
        row["messages"] = prefix + [user(text)] + exchange("read", {"record_id": e["id"]},
            {"record_id": e["id"], "state": e["initial_state"]}, "read-known") + [assistant(f"The state is {e['initial_state']}.")]
    elif kind in ("check_needed", "already_done"):
        # Identical task/schema/ID across the pair; the returned state alone
        # decides whether an update is needed.
        text = [f"Check record {e['id']} and set its state to {e['desired_state']} only if needed.",
                f"Ensure ID {e['id']} is {e['desired_state']}. Read its state first and avoid a redundant write.",
                f"Look up record {e['id']} by ID. If its state isn't {e['desired_state']}, update it.",
                f"Inspect {e['id']}; change its state to {e['desired_state']} only if it differs."][number % 4]
        if kind == "check_needed":
            e["initial_state"] = "pending"
        row["messages"] = prefix + [user(text)] + exchange("read", {"record_id": e["id"]},
            {"record_id": e["id"], "state": e["initial_state"]}, "inspect")
        if kind == "check_needed":
            row["messages"] += exchange("set_state", {"record_id": e["id"], "state": e["desired_state"]},
                {"status": "ok", "record_id": e["id"], "state": e["desired_state"]}, "update")
            row["messages"].append(assistant(f"Updated {e['id']} to {e['desired_state']}."))
        else:
            row["messages"].append(assistant(f"The record is already {e['desired_state']}. No change was needed."))
    else:
        # Change surface wording without changing the task or reference action.
        message = row["messages"][at]
        if number % 4 == 1:
            message["content"] = "Please " + message["content"][0].lower() + message["content"][1:]
        elif number % 4 == 2:
            message["content"] = message["content"].replace("Find the", "Locate the").replace("Set record", "Update record").replace(" and set its state to", "; its desired state is")
        elif number % 4 == 3:
            message["content"] = "Task: " + message["content"]
    # Vary API names and identifier argument names so the learner must read the
    # schema, not memorize *_search versus *_read or a particular tool order.
    variants = [("search", "read", "set_state", "refresh", "record_id"),
                ("find_by_name", "get_by_id", "update_state", "renew_session", "id"),
                ("locate", "inspect", "change_state", "restore_session", "key"),
                ("query_name", "fetch_record", "write_state", "refresh_session", "reference")]
    variant = variants[number % len(variants)]
    operations = ("search", "read", "set_state", "refresh")
    names = {domain + "_" + old: domain + "_" + new for old, new in zip(operations, variant[:4])}
    for t in row["tools"]:
        f = t["function"]
        f["name"] = names[f["name"]]
        properties = f["parameters"]["properties"]
        if "record_id" in properties:
            properties[variant[4]] = properties.pop("record_id")
            f["parameters"]["required"] = [variant[4] if p == "record_id" else p for p in f["parameters"]["required"]]
    for i, m in enumerate(row["messages"]):
        for c in m.get("tool_calls", []):
            f = c["function"]
            f["name"] = names[f["name"]]
            if "record_id" in f["arguments"]:
                f["arguments"][variant[4]] = f["arguments"].pop("record_id")
            m["content"] = ("" if (number + i) % 2 else
                            ["I'll do that now.", "I'll check using the available tool.", "I'll make that call."][(number + i) % 3])
        if m["role"] == "tool":
            m["name"] = names[m["name"]]
    random.Random(f"tools-v2/{number}").shuffle(row["tools"])
    e["operations"] = {names[domain + "_" + op]: op for op in operations}
    e["id_argument"] = variant[4]
    row.update(doc_id=f"agentic-v2:{domain}:{kind}:{number}", kind=kind, curriculum_version=2)
    return row


def make_trajectory(domain, kind, number):
    """A finite environment with explicit observations and reference actions."""
    rng = random.Random(f"agentic-v1/{domain}/{kind}/{number}")
    name = f"{rng.choice(['Cedar', 'Willow', 'Juniper', 'Maple', 'Aspen'])} {rng.randrange(1000, 9999)}"
    ident, alternate = f"{domain[:3]}-{rng.randrange(100000, 499999)}", f"{domain[:3]}-{rng.randrange(500000, 999999)}"
    desired = rng.choice(["reviewed", "active", "ready"])
    current = rng.choice(["pending", "draft", "new"])
    functions = {
        "search": domain + "_search",
        "read": domain + "_read",
        "set": domain + "_set_state",
        "refresh": domain + "_refresh",
    }
    tools = [schema(functions["search"], "Find all records matching a name. Returns IDs, names, and sections.", {"name": "string"}),
             schema(functions["read"], "Read the current state of a record by its ID.", {"record_id": "string"}),
             schema(functions["set"], "Set the state of a record by its ID. A STALE_SESSION error requires refresh before retrying.", {"record_id": "string", "state": "string"}),
             schema(functions["refresh"], "Refresh a stale session using the recovery token from the error.", {"recovery_token": "string"})]
    messages = []
    # Vary whether the explicit agent policy is present; every target is unchanged.
    if number % 3:
        messages.append({"role": "system", "content": POLICY})
    state = current
    searched = False
    refreshed = False
    attempts = 0
    recovery = f"REC-{rng.randrange(10000, 99999)}"
    entries = [{"record_id": ident, "name": name, "section": "north"}]
    if kind in ("ask_choice", "disambiguate"):
        entries.append({"record_id": alternate, "name": name, "section": "south"})
    if kind == "empty_search":
        entries = []

    def user(text):
        messages.append({"role": "user", "content": text})

    def say(text):
        messages.append({"role": "assistant", "content": text})

    def act(operation, arguments, announcement=""):
        nonlocal state, searched, refreshed, attempts
        # Reference actions are executed by this local state machine, not fabricated
        # independently of their observations. Mutation requires a known unique ID.
        if operation == "search":
            assert arguments == {"name": name}
            searched = True
            result = {"matches": entries, "complete": True}
        elif operation == "read":
            assert arguments == {"record_id": ident}
            assert searched or kind in ("known_id", "recover", "already_done")
            result = {"record_id": ident, "state": state}
        elif operation == "refresh":
            assert attempts == 1 and arguments == {"recovery_token": recovery}
            refreshed = True
            result = {"status": "ok"}
        else:
            assert operation == "set" and arguments == {"record_id": ident, "state": desired}
            assert searched or kind in ("known_id", "recover", "announce")
            attempts += 1
            if kind == "recover" and not refreshed:
                result = {"error": "STALE_SESSION", "recovery_token": recovery}
            else:
                state = desired
                result = {"status": "ok", "record_id": ident, "state": state}
        invocation = call(functions[operation], arguments)
        invocation["id"] = f"c{len(messages)}"
        messages.append({"role": "assistant", "content": announcement, "tool_calls": [invocation]})
        messages.append({"role": "tool", "tool_call_id": invocation["id"], "name": functions[operation],
                         "content": json.dumps(result, sort_keys=True)})

    if kind == "no_call":
        user(f"Explain in one sentence what it means for a {domain} record to be reviewed. Do not change any records.")
        say("Reviewed means the record has been checked; it does not by itself mean the record is active or ready.")
    elif kind == "already_done":
        state = desired
        user(f"Check record {ident} and set its state to {desired} only if needed.")
        act("read", {"record_id": ident})
        say(f"Record {ident} is already {desired}; no update was needed.")
    elif kind == "read_after_search":
        user(f"Look up the {domain} record named {name} and tell me its current state.")
        act("search", {"name": name})
        act("read", {"record_id": ident})
        say(f"{name} is currently {current}.")
    else:
        if kind in ("known_id", "recover", "announce"):
            user(f"Set record {ident} to {desired}.")
        else:
            qualifier = " in the north section" if kind == "disambiguate" else ""
            user(f"Find the {domain} record named {name}{qualifier} and set its state to {desired}.")
            act("search", {"name": name}, "I'll look up the record first." if number % 2 else "")
        if kind == "empty_search":
            say(f"The search returned no record named {name}. Please provide a different name or the record ID.")
        else:
            if kind == "ask_choice":
                say(f"There are records named {name} in the north and south sections. Which section should I update?")
                user("The north section, please.")
            act("set", {"record_id": ident, "state": desired},
                "I'll update the record now." if kind == "announce" or number % 4 == 0 else "")
            if kind == "recover":
                act("refresh", {"recovery_token": recovery}, "I'll refresh the session and retry.")
                act("set", {"record_id": ident, "state": desired})
            assert state == desired
            say(f"Record {ident} is now {desired}.")
    return {"doc_id": f"agentic-v1:{domain}:{kind}:{number}", "domain": domain,
            "kind": kind, "tools": tools, "messages": messages,
            "expected_state": state,
            "environment": {"id": ident, "alternate": alternate, "name": name,
                            "initial_state": desired if kind == "already_done" else current,
                            "desired_state": desired, "recovery_token": recovery,
                            "entries": entries}}


def validate(row):
    from tool_tasks import call_problem
    definitions = {t["function"]["name"]: t["function"] for t in row["tools"]}
    pending = set()
    for m in row["messages"]:
        if m["role"] == "assistant":
            assert not pending, "assistant ran before receiving all results"
            for c in m.get("tool_calls", []):
                assert not call_problem(c, definitions), call_problem(c, definitions)
                pending.add(c["id"])
            if "I'll " in m.get("content", ""):
                assert m.get("tool_calls"), "announcement without action"
        elif m["role"] == "tool":
            assert m["tool_call_id"] in pending
            pending.remove(m["tool_call_id"])
    assert not pending


def build(args):
    import numpy as np
    from transformers import AutoTokenizer
    from distillkit.offline_cache import OfflineCacheWriter, tokenizer_vocab_hash
    from teacher_kl import assistant_tokens
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    reference = json.loads((args.reference / "manifest.json").read_text())
    assert tokenizer_vocab_hash(tok) == reference["tokenizer_vocab_hash"]
    if args.output.exists():
        raise ValueError("refuse to overwrite curriculum output")
    args.output.mkdir(parents=True)
    rows, counts, fixtures = [], Counter(), []
    marker = tok.encode("<|im_start|>assistant", add_special_tokens=False)
    close = tok.convert_tokens_to_ids("<|im_end|>")
    cache = args.output / "cache"
    with OfflineCacheWriter(cache, tokenizer_hash=reference["tokenizer_hash"],
            tokenizer_vocab_fingerprint=reference["tokenizer_vocab_hash"],
            anchor_layers=[], hidden_size=reference["hidden_size"], vocab_size=reference["vocab_size"],
            top_k=reference["top_k"], sequence_length=8192,
            metadata={"target_kind": "hard_labels_only", "required_objective": "assistant_only_ce",
                      "reserved_topk": "uniform placeholders; never use as teacher predictions",
                      "generator": "agentic_curriculum.py", "version": args.version}) as writer:
        domains = DOMAINS if args.version == 1 else DOMAINS[:6] + ["dispatch", "records"]
        for domain in domains:
            split = "train" if domain in DOMAINS[:6] else "eval"
            for kind in (KINDS if args.version == 1 else CONTRAST_KINDS):
                for n in range(args.per_kind if split == "train" else 2):
                    row = (make_trajectory if args.version == 1 else make_contrast)(domain, kind, n)
                    validate(row)
                    if args.version == 2:
                        from agentic_live_eval import Environment
                        env = Environment(row)
                        for m in row["messages"]:
                            if m["role"] != "assistant":
                                continue
                            text = m["content"] + "".join("<tool_call>" + json.dumps({"name": c["function"]["name"],
                                "arguments": c["function"]["arguments"]}) + "</tool_call>" for c in m.get("tool_calls", []))
                            env.respond(text)
                        assert env.success and env.done, row["doc_id"]
                    row["split"] = split
                    text = tok.apply_chat_template(row["messages"], tools=row["tools"], tokenize=False,
                                                   add_generation_prompt=False, enable_thinking=False)
                    ids = np.asarray(tok.encode(text, add_special_tokens=False), dtype=np.int64)
                    keep = assistant_tokens(ids, marker, close)
                    assert 0 < keep.sum() < len(ids)
                    row["token_sha256"] = hashlib.sha256(ids.astype("<u4").tobytes()).hexdigest()
                    row["tokens"], row["supervised_tokens"] = len(ids), int(keep[1:].sum())
                    counts[f"{split}_documents"] += 1
                    counts[f"{split}_tokens"] += len(ids)
                    counts[f"{split}_supervised_tokens"] += row["supervised_tokens"]
                    top = np.broadcast_to(np.arange(reference["top_k"], dtype=np.uint32), (len(ids), reference["top_k"]))
                    logp = np.full(top.shape, -math.log(reference["vocab_size"]), dtype=np.float16)
                    writer.append(row["doc_id"], ids, top, logp, split=split)
                    rows.append(row)
                    # Freeze every held-out action and factual final response. Asks
                    # remain in held-out NLL; live task success needs a separate grader.
                    if split == "eval":
                        for index, m in enumerate(row["messages"]):
                            if m["role"] != "assistant":
                                continue
                            is_question = kind in ("ask_choice", "empty_search") and (
                                "Which section" in m["content"] or "Please provide" in m["content"])
                            if not m.get("tool_calls") and not is_question and kind not in ("read_after_search", "known_read", "already_done", "no_call"):
                                continue
                            prefix = row["messages"][:index]
                            prompt = tok.apply_chat_template(prefix, tools=row["tools"], tokenize=False,
                                                             add_generation_prompt=True, enable_thinking=False)
                            case = dict(id=f"{row['doc_id']}:turn{index}", category=kind,
                                        source_id=row["doc_id"], messages=prefix, tools=row["tools"],
                                        prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                                        prompt_tokens=len(tok.encode(prompt, add_special_tokens=False)))
                            if m.get("tool_calls"):
                                case["expected_calls"] = m["tool_calls"]
                            elif kind == "no_call":
                                case["expected_calls"] = []
                            elif is_question:
                                case["required_text"] = (["north", "south"] if kind == "ask_choice" else ["record"])
                                case["require_question"] = True
                            else:
                                case["expected_value"] = row["expected_state"]
                            fixtures.append(case)
    (args.output / "trajectories.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (args.output / "heldout-frozen.json").write_text(json.dumps({"cases": fixtures}, indent=2), encoding="utf-8")
    hashes = {s: {r["token_sha256"] for r in rows if r["split"] == s} for s in ("train", "eval")}
    assert not hashes["train"] & hashes["eval"]
    summary = dict(counts, version=args.version, fixture_cases=len(fixtures), train_domains=domains[:6], eval_domains=domains[6:],
                   generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   limitation="Shared templates across splits; schema/domain transfer screen, not independent task-family generalization.")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=Path("scratch/dense_gr/merges-long1/u50"))
    parser.add_argument("--reference", type=Path, default=Path("../teacher-cache-frontier-tools"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-kind", type=int, default=12)
    parser.add_argument("--version", type=int, choices=[1, 2], default=1)
    build(parser.parse_args())
