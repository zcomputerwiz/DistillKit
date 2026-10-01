"""Run a JSONL of chat requests against an OpenAI-compatible endpoint (OpenRouter), in parallel.

Each input line: {"id": str, "messages": [...], ...any other request fields (max_tokens,
temperature, reasoning, response_format)}. Each output line: {"id", "content", "reasoning",
"usage", "finish_reason"} or {"id", "error"}. Output is appended and flushed per result, and ids
already answered are skipped, so an interrupted run resumes where it stopped. The key is read
from --key-file and never printed or logged.

    python scratch/frontier/frontier_batch.py --input requests.jsonl --output responses.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "stealth/space-bunny-alpha"
KEY_FILE = Path.home() / ".claude" / ".openrouter"


def read_key(path):
    text = Path(path).read_text(encoding="utf-8").strip()
    # Accept a bare key or a NAME=key line.
    return text.split("=", 1)[1].strip().strip('"') if "=" in text.splitlines()[0] else text.splitlines()[0].strip()


def call(request, key, model, endpoint, retries=8, timeout=600):
    body = {"model": model, **{k: v for k, v in request.items() if k != "id"}}
    data = json.dumps(body).encode("utf-8")
    for attempt in range(retries):
        req = urllib.request.Request(endpoint, data=data, headers={
            "Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                reply = json.loads(response.read().decode("utf-8"))
            if "error" in reply:
                raise RuntimeError(json.dumps(reply["error"])[:500])
            choice = reply["choices"][0]
            message = choice.get("message", {})
            return {"id": request["id"], "content": message.get("content"),
                    "reasoning": message.get("reasoning") or message.get("reasoning_content"),
                    "finish_reason": choice.get("finish_reason"), "usage": reply.get("usage")}
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:500]
            if error.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(min(120, 2 ** attempt) + random.random())
                continue
            return {"id": request["id"], "error": "HTTP %d: %s" % (error.code, detail)}
        except Exception as error:  # timeouts, resets, malformed replies
            if attempt < retries - 1:
                time.sleep(min(120, 2 ** attempt) + random.random())
                continue
            return {"id": request["id"], "error": repr(error)[:500]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--key-file", type=Path, default=KEY_FILE)
    parser.add_argument("--limit", type=int, default=None, help="only the first N requests")
    args = parser.parse_args()
    key = read_key(args.key_file)
    done = set()
    if args.output.exists():
        for line in open(args.output, encoding="utf-8"):
            row = json.loads(line)
            if "error" not in row:
                done.add(row["id"])
    requests = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()][:args.limit]
    todo = [r for r in requests if r["id"] not in done]
    print("%d requests, %d already answered, %d to run" % (len(requests), len(requests) - len(todo), len(todo)),
          flush=True)
    lock, tally, started = threading.Lock(), {"ok": 0, "error": 0, "prompt": 0, "completion": 0, "cost": 0.0}, time.time()
    with open(args.output, "a", encoding="utf-8") as out, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(call, r, key, args.model, args.endpoint) for r in todo]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            with lock:
                out.write(json.dumps(row) + "\n")
                out.flush()
                tally["error" if "error" in row else "ok"] += 1
                usage = row.get("usage") or {}
                tally["prompt"] += usage.get("prompt_tokens") or 0
                tally["completion"] += usage.get("completion_tokens") or 0
                tally["cost"] += usage.get("cost") or 0.0
                if index % 25 == 0 or index == len(futures):
                    print("%d/%d  ok %d  errors %d  tokens in %d out %d  cost $%.3f  %.0f s"
                          % (index, len(futures), tally["ok"], tally["error"], tally["prompt"],
                             tally["completion"], tally["cost"], time.time() - started), flush=True)
                if "error" in row and tally["error"] <= 3:
                    print("error on %s: %s" % (row["id"], row["error"][:300]), flush=True)


if __name__ == "__main__":
    main()
