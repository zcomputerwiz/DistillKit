"""Which teacher layers can be stored as unscaled float8_e4m3fn?

`capture_teacher` stores anchor hidden states as raw fp8 and *refuses* to capture if
any value exceeds the format's range:

    if not torch.isfinite(hidden).all() or hidden.abs().max() > torch.finfo(float8_e4m3fn).max:
        raise ValueError(f"Anchor {layer_index} overflows unscaled float8_e4m3fn")

float8_e4m3fn maxes out at **448**. Qwen-family models are known for "massive
activations" -- a handful of channels in the residual stream growing to hundreds or
thousands by the deep layers. So this is not a theoretical limit: picking a deep anchor
could abort a multi-hour capture on document one, and picking one that *just* fits on a
sample could abort hours in on an unlucky document.

Failing loudly is the right design; the point of this probe is to find out, before
committing to a long run, which anchors are safe and by how much headroom.

Reports per layer: max |h|, the 99.999th percentile, and how many values would clip.
Anchor indices are into `output.hidden_states`, which has num_layers+1 entries -- index
0 is the embedding output, so index i is the input to layer i.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max

PROMPTS = [
    "The lighthouse keeper counted the ships passing the harbour at dawn, and wrote each name in a ledger bound in salt-stained leather.",
    "def binary_search(items, target):\n    lo, hi = 0, len(items) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2",
    "In 1687 Newton published the Principia, in which he set out three laws of motion and a law of universal gravitation.",
    "臺灣的東部海岸線以陡峭的斷崖聞名，太平洋的浪直接拍打在岩壁上。",
    "Q: If a train leaves Chicago at 3pm travelling 60 mph, and another leaves St. Louis at 4pm travelling 75 mph, when do they meet?",
    "\n\n\n   ...!!!???   ***   \x00 \t\t  ",  # degenerate input: often provokes outliers
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="../teacher-hf")
    parser.add_argument("--int8", action="store_true", default=True)
    parser.add_argument("--no-int8", dest="int8", action="store_false")
    parser.add_argument("--max-memory", default='{"0":"21GiB","1":"21GiB","cpu":"60GiB"}')
    parser.add_argument("--report")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "max_memory": {(int(k) if k.isdigit() else k): v
                       for k, v in json.loads(args.max_memory).items()},
        "local_files_only": True,
    }
    if args.int8:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    print(f"loading teacher (int8={args.int8}) ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()
    n_layers = model.config.get_text_config().num_hidden_layers
    print(f"loaded: {n_layers} layers, devices {sorted({str(p.device) for p in model.parameters()})}")

    stats = {}
    for text in PROMPTS:
        ids = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
        with torch.inference_mode():
            out = model(**ids, output_hidden_states=True, use_cache=False, return_dict=True)
        for i, hidden in enumerate(out.hidden_states):
            h = hidden.float().abs()
            entry = stats.setdefault(i, {"max": 0.0, "clipped": 0, "total": 0, "p99999": 0.0})
            entry["max"] = max(entry["max"], h.max().item())
            entry["clipped"] += int((h > FP8_MAX).sum())
            entry["total"] += h.numel()
            entry["p99999"] = max(entry["p99999"],
                                  torch.quantile(h.flatten().float(), 0.99999).item())
        del out

    print(f"\nfloat8_e4m3fn max = {FP8_MAX}\n")
    print(f"{'anchor':>6s} {'max|h|':>12s} {'p99.999':>10s} {'clipped':>9s}  status")
    safe = []
    for i in sorted(stats):
        e = stats[i]
        ok = e["max"] <= FP8_MAX
        if ok:
            safe.append(i)
        flag = "OK" if ok else f"OVERFLOW ({e['clipped']}/{e['total']})"
        headroom = f"{FP8_MAX / max(e['max'], 1e-9):5.1f}x" if ok else ""
        if i % 4 == 0 or not ok or i in (0, n_layers):
            print(f"{i:6d} {e['max']:12.2f} {e['p99999']:10.3f} {e['clipped']:9d}  {flag} {headroom}")

    print(f"\nfp8-safe anchors: {len(safe)}/{len(stats)}")
    if safe:
        # Prefer well-separated anchors with real headroom, mid and late network.
        comfortable = [i for i in safe if stats[i]["max"] < FP8_MAX / 4]
        print(f"comfortable (>4x headroom): {len(comfortable)}")
        if comfortable:
            mid = comfortable[len(comfortable) // 2]
            late = comfortable[int(len(comfortable) * 0.8)]
            print(f"suggested --anchor {mid} --anchor {late}  "
                  f"(max|h| {stats[mid]['max']:.1f} and {stats[late]['max']:.1f})")
    unsafe = [i for i in sorted(stats) if stats[i]["max"] > FP8_MAX]
    if unsafe:
        print(f"UNSAFE anchors (would abort capture): {unsafe}")

    if args.report:
        json.dump({"fp8_max": FP8_MAX, "layers": n_layers,
                   "stats": {str(k): v for k, v in stats.items()},
                   "safe": safe, "unsafe": unsafe},
                  open(args.report, "w"), indent=2)
        print("wrote", args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
