# Assisted-by: Codex
"""Closed-loop evaluation in finite local environments, with no external tools.

Allows extra reads and valid alternative action orders; never rewards call count.
This is a narrow synthetic task-success screen, not a general agent benchmark.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from agentic_curriculum import make_trajectory
from tool_behavior_eval import parse_calls
from tool_tasks import call_problem


class Environment:
    def __init__(self, row):
        self.row = row
        self.env = row["environment"]
        self.state = self.env["initial_state"]
        self.refreshed = False
        self.calls = 0
        self.mutations = 0
        self.errors = []
        self.asked = False
        self.choice = row["kind"] != "ask_choice"
        self.searched = False
        self.read = False
        self.done = False
        self.success = False
        self.messages = []
        for message in row["messages"]:
            if message["role"] == "assistant":
                break
            self.messages.append(message)
        self.definitions = {t["function"]["name"]: t["function"] for t in row["tools"]}

    def execute(self, invocation):
        self.calls += 1
        error = call_problem(invocation, self.definitions)
        if error:
            self.errors.append("invalid_schema")
            return {"error": "INVALID_CALL"}
        f = invocation["function"]
        operation = f["name"][len(self.row["domain"]) + 1:]
        a, e, kind = f["arguments"], self.env, self.row["kind"]
        if operation == "search":
            self.searched = True
            return {"matches": e["entries"] if a["name"] == e["name"] else [], "complete": True}
        if operation == "refresh":
            if a["recovery_token"] != e["recovery_token"]:
                self.errors.append("invented_recovery_token")
                return {"error": "INVALID_TOKEN"}
            self.refreshed = True
            return {"status": "ok"}
        if a["record_id"] not in (e["id"], e["alternate"]) or kind == "empty_search":
            self.errors.append("unknown_id")
            return {"error": "NOT_FOUND"}
        if operation == "read":
            self.read = True
            return {"record_id": a["record_id"], "state": self.state if a["record_id"] == e["id"] else "draft"}
        if operation == "set_state":
            if kind in ("no_call", "read_after_search") or not self.choice or a["record_id"] != e["id"] or a["state"] != e["desired_state"]:
                self.errors.append("unrequested_mutation")
                return {"error": "UNREQUESTED_MUTATION"}
            if kind == "recover" and not self.refreshed:
                return {"error": "STALE_SESSION", "recovery_token": e["recovery_token"]}
            self.mutations += 1
            self.state = a["state"]
            return {"status": "ok", "record_id": e["id"], "state": self.state}
        raise ValueError("unhandled validated operation")

    def respond(self, text, truncated=False):
        kind, e = self.row["kind"], self.env
        try:
            calls = parse_calls(text, self.definitions)
        except (ValueError, TypeError, KeyError):
            self.errors.append("parse_error")
            calls = []
        if truncated:
            self.errors.append("truncated")
        if calls:
            for i, c in enumerate(calls):
                c["id"] = f"live-{len(self.messages)}-{i}"
            self.messages.append({"role": "assistant", "content": text.split("<tool_call>")[0], "tool_calls": calls})
            for c in calls:
                observation = self.execute(c)
                self.messages.append({"role": "tool", "tool_call_id": c["id"],
                                      "name": c["function"]["name"], "content": json.dumps(observation)})
            return
        self.messages.append({"role": "assistant", "content": text})
        ask = bool(re.search(r"\b(which|provide|tell|share|confirm)\b", text.lower()))
        if kind == "ask_choice" and not self.choice and self.searched and ask and "north" in text.lower() and "south" in text.lower():
            self.asked = self.choice = True
            self.messages.append({"role": "user", "content": "The north section, please."})
            return
        self.done = True
        if kind == "no_call":
            self.success = self.calls == 0 and "review" in text.lower()
        elif kind == "empty_search":
            self.success = self.searched and ask and self.mutations == 0
        elif kind == "read_after_search":
            self.success = self.searched and self.read and self.state in text and self.mutations == 0
        elif kind == "already_done":
            self.success = self.read and self.mutations == 0 and self.state in text
        else:
            self.success = self.state == e["desired_state"] and self.mutations == 1 and not ask
            if kind == "ask_choice":
                self.success &= self.asked
        self.success &= not self.errors


def run(args):
    import smoke_train
    import torch
    from transformers import AutoTokenizer
    from distillkit.models import Qwen35WidenedForCausalLM
    if args.output.exists():
        raise ValueError("refuse to overwrite live evaluation")
    rows = [json.loads(l) for l in args.data.read_text(encoding="utf-8").splitlines()]
    environments = []
    for row in rows:
        if row["split"] != "eval":
            continue
        # Regenerate the deterministic environment while checking that every frozen
        # training/evaluation message is identical. Metadata was added after freeze.
        fresh = make_trajectory(row["domain"], row["kind"], int(row["doc_id"].rsplit(":", 1)[1]))
        assert fresh["messages"] == row["messages"] and fresh["tools"] == row["tools"]
        environments.append(Environment(fresh))
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    tok.padding_side, tok.pad_token_id = "left", 248044
    model = Qwen35WidenedForCausalLM.from_pretrained(args.checkpoint, dtype=torch.bfloat16).to("cuda:0").eval()
    model.config.use_cache = True
    torch.manual_seed(0)
    for turn in range(8):
        active = [e for e in environments if not e.done]
        if not active:
            break
        for start in range(0, len(active), 8):
            batch = active[start:start + 8]
            prompts = [tok.apply_chat_template(e.messages, tools=e.row["tools"], tokenize=False,
                        add_generation_prompt=True, enable_thinking=False) for e in batch]
            inputs = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                    temperature=None, top_p=None, top_k=None, eos_token_id=[248044, 248046], pad_token_id=248044)
            for e, output in zip(batch, outputs):
                tokens = output[inputs["input_ids"].shape[1]:].tolist()
                stops = [i for i, t in enumerate(tokens) if t in (248044, 248046)]
                e.respond(tok.decode(tokens[:stops[0]] if stops else tokens, skip_special_tokens=False), not stops)
        print(f"turn {turn + 1}: {sum(e.done for e in environments)}/{len(environments)} finished", flush=True)
    records = [dict(id=e.row["doc_id"], kind=e.row["kind"], success=e.success and e.done,
                    calls=e.calls, mutations=e.mutations, errors=e.errors, messages=e.messages) for e in environments]
    args.output.write_text(json.dumps(dict(checkpoint=str(args.checkpoint), records=records), indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
