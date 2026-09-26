"""Batch completions from a llama-server, for the stock models the compiled loop does not serve.

Sends the already-rendered chat prompt as raw text to `/completion` (special tokens in it
are parsed as special), many requests in flight at once to fill the server's parallel
slots. Every sampling parameter is set explicitly -- the server's own defaults (top-k 40,
min-p 0.05, temperature 0.8) would otherwise leak in. Greedy is temperature 0.
"""
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def complete(url, prompt, n_predict, sampling, seed):
    if sampling is None:
        params = dict(temperature=0.0, top_k=1, top_p=1.0, min_p=0.0)
    else:
        temperature, top_p, top_k = sampling
        params = dict(temperature=temperature, top_p=top_p, top_k=top_k, min_p=0.0)
    body = dict(prompt=prompt, n_predict=n_predict, seed=seed, cache_prompt=False,
                repeat_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
                stream=False, **params)
    request = urllib.request.Request(url.rstrip("/") + "/completion",
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3600) as response:
        out = json.loads(response.read())
    tokens = int(out.get("tokens_predicted", 0))
    truncated = out.get("stop_type") == "limit" or tokens >= n_predict
    return out.get("content", ""), tokens, truncated


def complete_all(url, prompts, n_predict, sampling=None, seed=0, workers=32):
    """`(text, tokens, truncated)` for each prompt, in order. Each request has its own seed
    derived from `seed` and its index, so a run is reproducible and rows differ."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(complete, url, p, n_predict, sampling, seed * 100003 + i)
                   for i, p in enumerate(prompts)]
        return [f.result() for f in futures]
