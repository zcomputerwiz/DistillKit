"""Tool-use conversations from the frontier model, checked against their own schemas.

Requests (`build`): each asks for four conversations in one domain and one scenario type,
with OpenAI-style function schemas, the expected calls, simulated tool results and final
answers. Domains times scenario types times seeds sets the diversity; BFCL's own functions
are never used, so the benchmark stays clean. Scenario types follow what tool use needs:
one call, independent parallel calls, a call that needs an earlier result, no call at
all, a missing required detail (ask), a tool error (recover), a follow-up turn, and
computation handed to a Python tool -- basic arithmetic is drilled in the head,
anything heavier goes to code.

Checking (`verify`): every call names a declared tool, its arguments are a JSON object
with the required keys, every key is declared, and each value matches its declared type
and enum. A conversation with any failing call is dropped.

    python scratch/frontier/tool_tasks.py build --output ../capture-data/frontier/tools-requests.jsonl
    python scratch/frontier/tool_tasks.py verify --responses ../capture-data/frontier/tools-responses.jsonl \\
        --output ../capture-data/frontier/tools.jsonl
"""
import argparse
import itertools
import json
import random
import re
from collections import Counter
from pathlib import Path

DOMAINS = """weather and forecasts; calendar and scheduling; e-commerce orders and returns; read-only banking \
and budgeting; Kubernetes operations; git and code review; local file system; SQL database queries; data \
analysis on CSV files; flights and hotels; clinic appointments; smart home devices; music streaming; maps and \
routing; translation and dictionaries; CRM and sales leads; IT ticketing; CI/CD pipelines; cloud object \
storage; email drafting and search; spreadsheets; web search and page fetching; unit and currency conversion; \
recipes and nutrition; fitness tracking; library catalog; package registry (npm/PyPI); log search and \
observability; DNS and networking diagnostics; container registry; issue trackers; HR and leave requests; \
inventory and warehouse; shipping and package tracking; stock market data (read-only); news aggregation; \
academic paper search; job postings; real estate listings; event ticketing; restaurant reservations; ride \
hailing; parking; public transit; IoT sensor telemetry; energy usage monitoring; customer support knowledge \
base; document OCR and parsing; image metadata; video transcoding jobs; feature flags; A/B test results; \
password-manager-free account settings; school timetable; sports scores; weather alerts for agriculture; \
legal document search; geocoding; time zones and clocks; chemistry property lookup; math and statistics via a \
Python tool""".split("; ")

SCENARIOS = {
    "single": "the user's request needs exactly one tool call; then the assistant answers from its result",
    "parallel": "the request needs two or three independent calls, made together in one assistant turn",
    "sequential": "a second call needs a value only the first call's result provides",
    "no_call": "the request can be answered directly or is outside every tool; the assistant makes no call",
    "missing_info": "a required argument is missing and cannot be guessed; the assistant asks for it instead "
                    "of calling, the user supplies it, and then the assistant calls",
    "error": "the first call returns an error (bad argument, not found, rate limit); the assistant corrects "
             "the call or explains, without inventing results",
    "follow_up": "after a first exchange with calls, the user asks a follow-up that needs another call",
    "python": "the user asks for computation beyond simple mental arithmetic (statistics, compound interest, "
              "date arithmetic, unit-heavy physics); the assistant calls a `python` tool with code that prints "
              "the result, then answers from the output",
}

INSTRUCTIONS = """You write training conversations that teach a small assistant model to use tools well.

Domain: {domain}
Scenario: {scenario}

Write 4 different conversations for this domain and scenario. For each:
- "tools": 2 to 6 realistic function schemas in OpenAI format ({{"type": "function", "function": {{"name", \
"description", "parameters": JSON Schema object with "properties" and "required"}}}}), including some tools the \
conversation does not need. For the python scenario, include {{"name": "python", "parameters": {{"type": \
"object", "properties": {{"code": {{"type": "string"}}}}, "required": ["code"]}}}}.
- "messages": the conversation in OpenAI chat format: "user" messages; "assistant" messages with "content" \
(may be empty when only calling) and optionally "tool_calls": [{{"id", "type": "function", "function": \
{{"name", "arguments": a JSON object}}}}]; "tool" messages with "tool_call_id" and "content" (the simulated \
result, realistic and consistent). End with an assistant message that answers the user.
Arguments must satisfy the schemas exactly (required keys present, types and enums respected, no extra keys). \
Use concrete, varied, realistic values; vary user tone and length. Final answers use only what the tools \
returned. Do not use tools or examples from public benchmarks.

Return only JSON: {{"conversations": [{{"tools": [...], "messages": [...]}}]}}"""

TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def build(args):
    rng = random.Random(args.seed)
    pairs = list(itertools.product(DOMAINS, SCENARIOS)) * args.repeats
    rng.shuffle(pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for index, (domain, scenario) in enumerate(pairs):
            out.write(json.dumps({
                "id": "tools:%d:%s" % (index, scenario),
                "messages": [{"role": "user", "content": INSTRUCTIONS.format(domain=domain, scenario=SCENARIOS[scenario])}],
                "max_tokens": 16000, "temperature": 1.0, "reasoning": {"effort": "low"},
                "response_format": {"type": "json_object"}}) + "\n")
    print("%d requests (%d domains x %d scenarios x %d) -> %s"
          % (len(pairs), len(DOMAINS), len(SCENARIOS), args.repeats, args.output))


def value_ok(value, schema):
    kind = schema.get("type")
    if isinstance(kind, list):
        return any(value_ok(value, dict(schema, type=k)) for k in kind)
    if kind == "null":
        return value is None
    if kind in TYPES:
        if kind in ("integer", "number") and isinstance(value, bool):
            return False
        if not isinstance(value, TYPES[kind]):
            return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if kind == "array" and isinstance(schema.get("items"), dict):
        return all(value_ok(v, schema["items"]) for v in value)
    if kind == "object" and isinstance(schema.get("properties"), dict):
        return all(k in schema["properties"] and value_ok(v, schema["properties"][k]) for k, v in value.items()) \
            and all(k in value for k in schema.get("required", []))
    return True


def call_problem(call, tools):
    function = (call or {}).get("function") or {}
    tool = tools.get(function.get("name"))
    if tool is None:
        return "unknown tool"
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return "arguments not JSON"
    if not isinstance(arguments, dict):
        return "arguments not an object"
    parameters = tool.get("parameters") or {}
    properties = parameters.get("properties") or {}
    if any(k not in properties for k in arguments):
        return "undeclared argument"
    if any(k not in arguments for k in parameters.get("required", [])):
        return "missing required argument"
    if not all(value_ok(v, properties[k]) for k, v in arguments.items()):
        return "argument type or enum"
    call["function"]["arguments"] = arguments
    return None


def conversation_problem(conversation, scenario):
    tools = {t.get("function", {}).get("name"): t.get("function", {}) for t in conversation.get("tools") or []
             if isinstance(t, dict)}
    messages = conversation.get("messages") or []
    if not tools or not messages or messages[0].get("role") != "user" or messages[-1].get("role") != "assistant":
        return "shape"
    calls, ids = 0, set()
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                problem = call_problem(call, tools)
                if problem:
                    return problem
                calls += 1
                ids.add(call.get("id"))
        elif message.get("role") == "tool" and message.get("tool_call_id") not in ids:
            return "tool result without a matching call"
    if scenario == "no_call" and calls:
        return "no_call scenario made a call"
    if scenario != "no_call" and not calls:
        return "no call made"
    if not str(messages[-1].get("content") or "").strip():
        return "empty final answer"
    return None


def verify(args):
    outcomes, kept = Counter(), Counter()
    with open(args.output, "w", encoding="utf-8") as out:
        for row in map(json.loads, open(args.responses, encoding="utf-8")):
            scenario = row["id"].rsplit(":", 1)[1]
            try:
                conversations = json.loads(re.sub(r"^```(json)?|```$", "", (row.get("content") or "").strip()))["conversations"]
            except (ValueError, KeyError, TypeError):
                outcomes["unparsable or failed"] += 1
                continue
            for index, conversation in enumerate(conversations):
                problem = conversation_problem(conversation, scenario) if isinstance(conversation, dict) else "shape"
                outcomes[problem or "ok"] += 1
                if problem is None:
                    kept[scenario] += 1
                    out.write(json.dumps({"doc_id": "%s:%d" % (row["id"], index), "scenario": scenario,
                                          "tools": conversation["tools"], "messages": conversation["messages"]}) + "\n")
    print("outcomes: %s" % dict(outcomes.most_common()))
    print("kept %d conversations: %s" % (sum(kept.values()), dict(kept)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--repeats", type=int, default=3)
    b.add_argument("--seed", type=int, default=0)
    v = sub.add_parser("verify")
    v.add_argument("--responses", type=Path, required=True)
    v.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (build if args.command == "build" else verify)(args)


if __name__ == "__main__":
    main()
