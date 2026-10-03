"""Aggregate and single-stream decode speed of a llama.cpp server: N concurrent fixed prompts."""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

server, concurrent, tokens = "http://127.0.0.1:8090", int(sys.argv[1]), int(sys.argv[2])
prompt = "<|im_start|>user\nWrite a detailed explanation of how a hash map works, with examples.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def one(seed):
    body = json.dumps({"prompt": prompt, "n_predict": tokens, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                       "seed": seed, "ignore_eos": True}).encode()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(server + "/completion", body,
                                                                  {"Content-Type": "application/json"})).read())
    return r["tokens_predicted"], r["timings"]["predicted_per_second"], r["timings"].get("draft_n_accepted"), r["timings"].get("draft_n")


start = time.time()
with ThreadPoolExecutor(concurrent) as pool:
    rows = list(pool.map(one, range(concurrent)))
wall = time.time() - start
total = sum(r[0] for r in rows)
print("concurrent %d: aggregate %.0f tok/s, per stream %.1f tok/s%s" % (
    concurrent, total / wall, sum(r[1] for r in rows) / len(rows),
    "" if rows[0][3] is None else ", draft accepted %d/%d" % (sum(r[2] or 0 for r in rows), sum(r[3] or 0 for r in rows))))
