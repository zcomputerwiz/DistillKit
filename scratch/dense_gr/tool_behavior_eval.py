# Assisted-by: Codex
"""Frozen, offline tool behavior screen using existing model generation and validators.

No generated tools or code are executed. Synthetic controlled pairs supplement
held-out next-call replay; this is not BFCL, API-Bank, or end-to-end task success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scratch/frontier"))
sys.path.insert(0, str(ROOT / "scratch/downstream/code_bench"))
from tool_tasks import call_problem


def schema(name, description, fields):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": {
                k: {"type": v} for k, v in fields.items()}, "required": list(fields)}}}


def call(name, arguments):
    return {"id": "fixture_call", "type": "function",
            "function": {"name": name, "arguments": arguments}}


def parse_calls(text, tools=None):
    """Strict completed Qwen XML calls or a completed JSON tool-call block."""
    out = []
    blocks = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.S)
    for block in blocks:
        match = re.fullmatch(r"\s*<function=([^>]+)>(.*?)</function>\s*", block, re.S)
        if match:
            name, body = match.groups()
            args = {}
            spans = list(re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", body, re.S))
            remainder = re.sub(r"<parameter=[^>]+>.*?</parameter>", "", body, flags=re.S)
            if remainder.strip():
                raise ValueError("unparsed function body")
            for span in spans:
                key, value = span.groups()
                if key in args:
                    raise ValueError("duplicate parameter")
                value = value.strip()
                declared = ((tools or {}).get(name.strip(), {}).get("parameters", {})
                            .get("properties", {}).get(key, {}).get("type"))
                try:
                    decoded = json.loads(value)
                    args[key] = value if declared == "string" and not isinstance(decoded, str) else decoded
                except json.JSONDecodeError:
                    args[key] = value
            out.append(call(name.strip(), args))
        else:
            obj = json.loads(block)
            out.append(call(obj["name"], obj["arguments"]))
    if text.count("<tool_call>") != len(blocks):
        raise ValueError("incomplete tool call")
    return out


def score(case, text, truncated=False):
    try:
        tools = {t["function"]["name"]: t["function"] for t in case["tools"]}
        calls = parse_calls(text, tools)
        problems = [call_problem(c, tools) for c in calls]
        malformed = False
    except (ValueError, KeyError, TypeError):
        calls, problems, malformed = [], ["parse error"], True
    want = case.get("expected_calls")
    canonical = lambda cs: sorted(json.dumps(c["function"], sort_keys=True) for c in cs)
    exact = (not malformed and not any(problems) and
             canonical(calls) == canonical(want)) if want is not None else None
    value = case.get("expected_value")
    grounded = (not calls and not malformed and value in text and
                all(v not in text for v in case.get("forbidden_values", []))) if value else None
    no_call = not calls and not malformed
    if case["category"] == "missing_info":
        success = no_call and "ticket number" in text.lower() and bool(
            re.search(r"\b(provide|tell|share|what|which|confirm)\b", text.lower()))
    elif value:
        success = grounded
    else:
        success = exact
    return {"success": bool(success) and not truncated, "exact_calls": exact,
            "grounded_value": grounded, "no_call": no_call,
            "schema_valid": bool(calls) and not any(problems), "parse_error": malformed,
            "truncated": truncated, "calls": calls, "problems": problems}


def fixtures():
    rng = random.Random(20261006)
    tools = [schema("lookup_ticket", "Find a ticket by its public ticket number.", {"ticket_number": "string"}),
             schema("get_ticket_details", "Read full details using the internal ticket ID returned by lookup_ticket.",
                    {"ticket_id": "string"})]
    cases = []
    for kind in ("extract", "dependent", "error", "long_extract"):
        for index in range(4):
            values = ["OBS_%s_%d" % (kind, rng.randrange(100000, 999999)) for _ in range(2)]
            for variant, value in enumerate(values):
                active_tools = tools
                user = "Look up ticket PUBLIC-%d and report its status code exactly as returned. Do not fetch extra details." % index
                expected = dict(expected_value=value, forbidden_values=[values[1 - variant]])
                output = {"ticket_id": "internal-%d" % index, "status_code": value}
                if kind == "dependent":
                    user = "Look up ticket PUBLIC-%d, then fetch its full details using the internal ID returned by lookup_ticket." % index
                    output = {"ticket_id": value}
                    expected = dict(expected_calls=[call("get_ticket_details", {"ticket_id": value})])
                if kind == "error":
                    active_tools = tools + [schema("refresh_token", "Refresh using an expired request's renewal key.",
                                                   {"renewal_key": "string"}),
                                             schema("schedule_retry", "Schedule a retry after the returned delay.",
                                                    {"retry_after_seconds": "integer"})]
                    user = ("Look up ticket PUBLIC-%d. If lookup returns EXPIRED, call refresh_token with its renewal_key. "
                            "If it returns RATE_LIMIT, call schedule_retry with its retry_after_seconds. Do not guess.") % index
                    output = ({"error": "EXPIRED", "renewal_key": value} if variant == 0 else
                              {"error": "RATE_LIMIT", "retry_after_seconds": 31 + index})
                    expected = dict(expected_calls=[call("refresh_token", {"renewal_key": value})] if variant == 0 else
                                    [call("schedule_retry", {"retry_after_seconds": 31 + index})])
                if kind == "long_extract":
                    output["unrelated_log"] = ["diagnostic record %d: health=normal; sample=unrelated" % n
                                               for n in range(140 + index * 35)]
                messages = [{"role": "user", "content": user},
                            {"role": "assistant", "content": "", "tool_calls": [call("lookup_ticket", {"ticket_number": "PUBLIC-%d" % index})]},
                            {"role": "tool", "tool_call_id": "fixture_call", "name": "lookup_ticket",
                             "content": json.dumps(output)}]
                cases.append(dict(id="%s-%d-%d" % (kind, index, variant), pair="%s-%d" % (kind, index),
                                  category=kind, tools=active_tools, messages=messages, **expected))
    for index in range(8):
        number = "PUBLIC-%d" % rng.randrange(10000, 99999)
        cases.append(dict(id="initial-%d" % index, category="initial", tools=tools,
                          messages=[{"role": "user", "content": "Find ticket %s." % number}],
                          expected_calls=[call("lookup_ticket", {"ticket_number": number})]))
    for index in range(4):
        cases.append(dict(id="missing-%d" % index, category="missing_info", tools=tools,
                          messages=[{"role": "user", "content": "Please fetch the full details of my ticket. I have not given you its number yet."}]))
        cases.append(dict(id="no-call-%d" % index, category="no_call", tools=tools,
                          messages=[{"role": "user", "content": "Reply with exactly ACK-%d. No ticket lookup is needed." % index}],
                          expected_value="ACK-%d" % index))
    return cases


def build(args):
    from transformers import AutoTokenizer
    from distillkit.offline_cache import OfflineTeacherCache

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    cases = [] if args.post_tool_only else fixtures()
    cache = OfflineTeacherCache(args.cache)
    train_ids, eval_ids = set(cache.document_ids("train")), set(cache.document_ids("eval"))
    pool = []
    for line in args.source.read_text(encoding="utf-8").splitlines():
        doc = json.loads(line)
        ident = doc.get("doc_id", doc.get("id"))
        ident = "tools:" + ident
        if ident not in eval_ids or ident in train_ids:
            continue
        for at, msg in enumerate(doc["messages"]):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                if args.post_tool_only and not any(m.get("role") == "tool" for m in doc["messages"][:at]):
                    continue
                pool.append((ident, at, doc))
    random.Random(20261006).shuffle(pool)
    used = set()
    for ident, at, doc in pool:
        if ident in used:
            continue
        used.add(ident)
        history = doc["messages"][:at]
        calls = doc["messages"][at]["tool_calls"]
        for c in calls:
            if isinstance(c["function"]["arguments"], str):
                c["function"]["arguments"] = json.loads(c["function"]["arguments"])
        cases.append(dict(id="replay:%s:%d" % (ident, at), category="heldout_call",
                          source_id=ident, tools=doc["tools"], messages=history, expected_calls=calls))
        if len(used) >= args.replay:
            break
    if not used and args.replay:
        raise ValueError("no held-out source IDs matched the cache")
    for case in cases:
        prompt = tok.apply_chat_template(case["messages"], tools=case["tools"], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        case.update(prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    prompt_tokens=len(tok.encode(prompt, add_special_tokens=False)))
        if case["prompt_tokens"] > 8192:
            raise ValueError("fixture exceeds screen's 8K input cap")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError("refuse to overwrite frozen fixture")
    payload = dict(format_version=1, seed=20261006, cases=cases, replay_documents=len(used),
                   train_id_exclusion=True, cache_manifest_sha256=hashlib.sha256(
                       (args.cache / "manifest.json").read_bytes()).hexdigest(),
                   source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                   note="Synthetic pairs are new fixtures; replay uses cache eval IDs, one next-call target per document. No execution.")
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(cases), "replay_documents": len(used),
                      "max_prompt_tokens": max(c["prompt_tokens"] for c in cases)}), flush=True)


def framed(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    cases = {}
    for name in ("frozen.json", "post-tool-frozen.json"):
        data = json.loads((args.directory / name).read_text(encoding="utf-8"))
        for case in data["cases"]:
            if case["category"] in ("heldout_call", "error"):
                cases[case["id"]] = case
    instruction = ("You are an assistant with access to the supplied tools. Use them when needed to fulfill "
                   "the user's request. Invoke a tool by emitting its native tool-call format; describing "
                   "an intended call does not execute it. Do not claim a supplied tool is unavailable. "
                   "Use returned values for subsequent calls. Ask for genuinely missing required information "
                   "rather than guessing.")
    for case in cases.values():
        case["messages"] = [{"role": "system", "content": instruction}] + case["messages"]
        prompt = tok.apply_chat_template(case["messages"], tools=case["tools"], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        case.update(prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    prompt_tokens=len(tok.encode(prompt, add_special_tokens=False)))
    if args.output.exists():
        raise ValueError("refuse to overwrite frozen prompt ablation")
    args.output.write_text(json.dumps(dict(cases=list(cases.values()), system_instruction=instruction,
        note="Exploratory prompt ablation selected after the primary screen; not an independent confirmation.",
        parent_hashes={name: hashlib.sha256((args.directory/name).read_bytes()).hexdigest()
                       for name in ("frozen.json", "post-tool-frozen.json")}), indent=2), encoding="utf-8")
    print("%d framed cases" % len(cases))


def run(args):
    import smoke_train
    import torch
    from transformers import AutoTokenizer
    from distillkit.models import Qwen35WidenedForCausalLM
    from generate import CompiledGreedy

    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    cases = fixture["cases"]
    if args.limit:
        cases = cases[:args.limit]
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    tok.padding_side = "left"
    tok.pad_token_id = 248044
    for case in cases:
        rendered = tok.apply_chat_template(case["messages"], tools=case["tools"], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
        if rendered != case["prompt"]:
            raise ValueError("checkpoint template changes frozen prompt")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.json").exists():
        raise ValueError("refuse to overwrite completed evaluation")
    torch.manual_seed(0)
    model = Qwen35WidenedForCausalLM.from_pretrained(args.checkpoint, dtype=torch.bfloat16).to(args.device).eval()
    model.config.use_cache = True
    records, started = [], time.monotonic()
    # Separate long fixtures so short ones are not padded to 8K. Same shapes across arms.
    for long in (False, True):
        subset = [c for c in cases if (c["category"] == "long_extract") == long]
        if not subset:
            continue
        width = ((max(c["prompt_tokens"] for c in subset) + 63) // 64) * 64
        runner = CompiledGreedy(model, args.batch, width, args.new, 248046) if args.compiled else None
        for start in range(0, len(subset), args.batch):
            batch = subset[start:start + args.batch]
            prompts = [c["prompt"] for c in batch]
            if runner:
                prompts += [prompts[0]] * (args.batch - len(prompts))
            inputs = tok(prompts, return_tensors="pt", padding="max_length" if runner else True,
                         max_length=width if runner else None, add_special_tokens=False).to(args.device)
            with torch.inference_mode():
                outputs = (runner(inputs["input_ids"], inputs["attention_mask"]) if runner else
                           model.generate(**inputs, max_new_tokens=args.new, do_sample=False,
                                          temperature=None, top_p=None, top_k=None,
                                          eos_token_id=[248044, 248046], pad_token_id=248044))
            for index, case in enumerate(batch):
                new = outputs[index, inputs["input_ids"].shape[1]:].tolist()
                stops = [n for n, t in enumerate(new) if t in (248044, 248046)]
                length = stops[0] if stops else len(new)
                text = tok.decode(new[:length], skip_special_tokens=False)
                records.append(dict(id=case["id"], category=case["category"], pair=case.get("pair"),
                                    prompt_sha256=case["prompt_sha256"], prompt_tokens=case["prompt_tokens"],
                                    raw=text, generated_tokens=length,
                                    **score(case, text, not stops and length >= args.new)))
            (args.output / "progress.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
            print("%s %d/%d %.1fs" % (args.arm, len(records), len(cases), time.monotonic() - started), flush=True)
        del runner
        torch.cuda.empty_cache()
    payload = dict(arm=args.arm, checkpoint=str(args.checkpoint.resolve()),
                   config_sha256=hashlib.sha256((args.checkpoint / "config.json").read_bytes()).hexdigest(),
                   fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
                   device=args.device, max_new_tokens=args.new, batch=args.batch, compiled=args.compiled,
                   stops=[248044, 248046], elapsed_seconds=time.monotonic() - started, records=records)
    (args.output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def report(args):
    import numpy as np
    fixtures_by_hash = {}
    for path in args.directory.glob("*frozen.json"):
        name = path.name
        fixtures_by_hash[hashlib.sha256(path.read_bytes()).hexdigest()] = (
            name, json.loads(path.read_text(encoding="utf-8"))["cases"])
    sets, checkpoints = {}, {}
    for path in sorted(args.directory.glob("*/results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["arm"] not in ("base", "long5", "ramp", "flat", "context"):
            continue
        name, cases = fixtures_by_hash[payload["fixture_sha256"]]
        lookup = {c["id"]: c for c in cases}
        if set(lookup) != {r["id"] for r in payload["records"]}:
            raise ValueError("missing/extra cases in " + str(path))
        records = []
        for original in payload["records"]:
            case = lookup[original["id"]]
            if original["prompt_sha256"] != case["prompt_sha256"]:
                raise ValueError("prompt hash mismatch")
            records.append(dict(original, **score(case, original["raw"], original["truncated"])))
        # Rescore every arm identically after correcting punctuation-only ask grading
        # and allowing independent calls to be emitted in a different order.
        graded = dict(payload, grader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), records=records)
        path.with_name("graded.json").write_text(json.dumps(graded, indent=2), encoding="utf-8")
        sets.setdefault(name, {})[payload["arm"]] = (lookup, records)
        checkpoints[payload["arm"]] = payload["checkpoint"]
    summaries = {}
    for name, arms in sets.items():
        if set(arms) != {"base", "long5", "ramp", "flat", "context"}:
            raise ValueError("all five checkpoints must complete for " + name)
        categories = sorted({c["category"] for c in arms["base"][0].values()})
        summary = {}
        for arm, (lookup, records) in arms.items():
            row = {}
            base = {r["id"]: r for r in arms["base"][1]}
            for category in categories:
                subset = [r for r in records if r["category"] == category]
                groups = {}
                for r in subset:
                    c = lookup[r["id"]]
                    unit = c.get("pair", c.get("source_id", c["prompt_sha256"]))
                    groups.setdefault(unit, []).append(r)
                values = np.array([[sum(int(r["success"]) - int(base[r["id"]]["success"]) for r in group),
                                    len(group)] for group in groups.values()])
                rng = np.random.default_rng(0)
                samples = rng.integers(0, len(values), (2000, len(values)))
                totals = values[samples].sum(axis=1)
                ci = np.percentile(totals[:, 0] / totals[:, 1], [2.5, 97.5]).tolist()
                row[category] = dict(n=len(subset), units=len(groups), success=sum(r["success"] for r in subset),
                                     emitted_calls=sum(bool(r["calls"]) for r in subset),
                                     schema_valid=sum(r["schema_valid"] for r in subset),
                                     truncated=sum(r["truncated"] for r in subset),
                                     delta_vs_base=sum(v[0] for v in values)/len(subset), paired_95_ci=ci,
                                     learned=sum(r["success"] and not base[r["id"]]["success"] for r in subset),
                                     forgot=sum(not r["success"] and base[r["id"]]["success"] for r in subset))
            pairs = {}
            for r in records:
                if r.get("pair"):
                    pairs.setdefault(r["pair"], []).append(r["success"])
            row["both_pair_variants"] = {"n": len(pairs), "success": sum(all(v) for v in pairs.values())}
            summary[arm] = row
        summaries[name] = summary
    fingerprints = {}
    for arm, directory in checkpoints.items():
        weights = {}
        for path in sorted(Path(directory).glob("*.safetensors")):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            weights[path.name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
        fingerprints[arm] = {"path": directory, "weights": weights}
    args.output.write_text(json.dumps({"sets": summaries, "checkpoint_fingerprints": fingerprints,
        "fixture_hashes": {name: digest for digest, (name, _) in fixtures_by_hash.items()},
        "grading_note": "All raw outputs rescored with the same grader. Exact replay call match is not execution/semantic equivalence. Missing-info grade is a request heuristic. Synthetic pair/duplicate-prompt groups bootstrap jointly; heldout units are documents. Small screen, no full-task success or sampling reliability."}, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--tokenizer", type=Path, required=True)
    b.add_argument("--cache", type=Path, required=True)
    b.add_argument("--source", type=Path, required=True)
    b.add_argument("--replay", type=int, default=24)
    b.add_argument("--post-tool-only", action="store_true")
    b.add_argument("--output", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--fixture", type=Path, required=True)
    r.add_argument("--checkpoint", type=Path, required=True)
    r.add_argument("--arm", required=True)
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--batch", type=int, default=8)
    r.add_argument("--new", type=int, default=512)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--compiled", action="store_true")
    r.add_argument("--output", type=Path, required=True)
    summary = sub.add_parser("report")
    summary.add_argument("--directory", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    framing = sub.add_parser("framed")
    framing.add_argument("--directory", type=Path, required=True)
    framing.add_argument("--tokenizer", type=Path, required=True)
    framing.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    {"build": build, "run": run, "report": report, "framed": framed}[args.command](args)


if __name__ == "__main__":
    main()
