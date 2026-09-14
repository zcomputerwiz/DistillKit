"""Is generating the codes cheaper than looking them up, in the path that actually runs?

Quality equivalence would not be enough. A table-free basis that costs real time in the
integrated forward is a worse engineering answer than a 9 MB buffer, so this times the
whole thing -- backbone, hash, code construction, decoder, logit bias -- rather than the
code construction on its own, which would measure an operation nobody runs in isolation.

Interleaved rounds and the minimum per arm, for the reason the residual-gate overhead
probe had to learn: the boost clock moves more between two back-to-back measurements of
identical work than these arms differ from each other.

    CUDA_VISIBLE_DEVICES=0 python scratch/structural_sidecar/benchmark.py \
        --output scratch/structural_sidecar/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffn_memo"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffn_attenuation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "residual_gate"))

import torch

from attenuate import load_records
from distillkit.experimental.ngram_hash import NGramHashConfig, NGramHasher
from distillkit.experimental.structural_sidecar import (
    StructuralSidecar, apply_structural_bias, structural_token_ids)
from evaluate import split_layout
from fit import BACKBONES, BASE
from repeatability import DEFAULT_BUNDLE

ARMS = {"fixed": Path("scratch/structural_sidecar/fixed-B42/sidecar.pt"),
        "direct": Path("scratch/structural_sidecar/direct-B42/sidecar.pt")}


def build(saved, structural, device):
    sidecar = StructuralSidecar(rows=saved["rows"], code_dim=saved["code_dim"],
                                structural=int(structural.numel()), mode=saved["mode"],
                                heads=saved["heads"], hidden=saved["hidden"],
                                seed=saved["seed"]).to(device).to(torch.float32)
    sidecar.load_state_dict(saved["state_dict"])
    sidecar.requires_grad_(False)
    return sidecar


@torch.inference_mode()
def run(model, sidecar, hasher, structural, tokens, strength, repeats):
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        logits = model(input_ids=tokens,
                       attention_mask=torch.ones_like(tokens)).logits[:, :-1].float()
        bias = sidecar(hasher.row_indices(tokens))[:, :-1]
        apply_structural_bias(logits, bias, structural, strength)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / repeats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="B42", choices=sorted(BACKBONES))
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise SystemExit("mask to one card with CUDA_VISIBLE_DEVICES")

    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from distillkit.independent_eval import build_token_classes

    checkpoint = BACKBONES[args.backbone]
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.use_cache = False
    model = Qwen3_5ForCausalLM.from_pretrained(
        checkpoint, config=config, dtype=torch.bfloat16,
        local_files_only=True).to(args.device).eval()
    model.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    classes = split_layout(build_token_classes(tokenizer, config.vocab_size), tokenizer,
                           config.vocab_size)
    structural = structural_token_ids(classes, device=args.device)
    record = load_records(args.bundle, "screen", 1)[0]
    tokens = torch.tensor([record["ids"]], device=args.device)

    built = {}
    for name, path in ARMS.items():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        sidecar = build(saved, structural, args.device)
        hasher = NGramHasher(NGramHashConfig(
            vocab_size=config.vocab_size, ngram_size=3, heads_per_ngram=1,
            ngram_vocab_size_base=saved["rows"] // 2 - 64, seed=1234,
            eos_token_id=tokenizer.eos_token_id or config.vocab_size - 1))
        built[name] = (sidecar, hasher, saved["strength"], saved)

    for name, (sidecar, hasher, strength, _) in built.items():
        run(model, sidecar, hasher, structural, tokens, strength, 2)

    report = {"backbone": args.backbone, "tokens": int(tokens.numel()),
              "rounds": args.rounds, "repeats": args.repeats, "arms": {}}
    samples = {name: [] for name in built}
    for _ in range(args.rounds):
        for name, (sidecar, hasher, strength, _) in built.items():
            samples[name].append(
                run(model, sidecar, hasher, structural, tokens, strength, args.repeats))

    for name, (sidecar, hasher, strength, saved) in built.items():
        torch.cuda.reset_peak_memory_stats(args.device)
        run(model, sidecar, hasher, structural, tokens, strength, 2)
        values = sorted(samples[name])
        report["arms"][name] = {
            "seconds_per_forward": values[0],
            "median_seconds": values[len(values) // 2],
            "spread_fraction": values[-1] / values[0] - 1.0,
            "tokens_per_second": int(tokens.numel() / values[0]),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(args.device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(args.device)),
            "code_table_bytes": sum(b.numel() * b.element_size()
                                    for b in sidecar.buffers()),
            "checkpoint_bytes": ARMS[name].stat().st_size,
            "trainable_parameters": sum(p.numel() for p in sidecar.parameters()),
        }
        entry = report["arms"][name]
        print("%-8s %.5f s/forward (median %.5f, spread %.1f%%)  %d tok/s  "
              "code table %d B  peak %.2f GiB"
              % (name, entry["seconds_per_forward"], entry["median_seconds"],
                 100 * entry["spread_fraction"], entry["tokens_per_second"],
                 entry["code_table_bytes"],
                 entry["peak_allocated_bytes"] / 2 ** 30), flush=True)

    fixed = report["arms"]["fixed"]["seconds_per_forward"]
    direct = report["arms"]["direct"]["seconds_per_forward"]
    report["direct_over_fixed"] = direct / fixed - 1.0
    print("direct is %+.2f%% against fixed; the larger arm spread is %.1f%%"
          % (100 * report["direct_over_fixed"],
             100 * max(report["arms"][name]["spread_fraction"] for name in report["arms"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
