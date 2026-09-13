"""What the gate actually costs, measured so the answer survives clock drift.

The first attempt at this reported +7%, +14% and +28% for three gates whose feature
counts go the other way, with the stock reference itself moving 9% between back-to-back
measurements. That is a measurement of the GPU's boost clock, not of the gate.

So: alternate gated and stock timings inside one process, several rounds each, and take
the minimum per arm. The minimum of repeated timings of identical work is the run least
disturbed by everything else on the machine, which is the number that answers "does this
cost anything" without pretending the mean of a contaminated sample means something.

    CUDA_VISIBLE_DEVICES=0 python scratch/residual_gate/overhead.py --output ...
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

import torch

from attenuate import load_records
from distillkit.residual_gate import (TrigramFamiliarity, family_features,
                                      install_residual_gates, remove_residual_gates)
from repeatability import DEFAULT_BUNDLE, DEFAULT_MODEL

CACHE = Path("scratch/ffn_memo/cache/layer-12.npz")


def measure(model, ids, device, repeats=20):
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(repeats):
            model(input_ids=ids, attention_mask=torch.ones_like(ids))
    torch.cuda.synchronize(device)
    return (time.perf_counter() - started) / repeats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--gates", nargs="+", required=True,
                        help="gate checkpoints to time, one per family")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise SystemExit("mask to one card with CUDA_VISIBLE_DEVICES")

    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.use_cache = False
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.model, config=config, dtype=torch.bfloat16,
        local_files_only=True).to(args.device).eval()
    ids = torch.tensor([load_records(args.bundle, "screen", 1)[0]["ids"]],
                       device=args.device)

    report = {"model": args.model, "tokens": int(ids.numel()), "rounds": args.rounds,
              "arms": {}}
    for _ in range(2):
        measure(model, ids, args.device, repeats=5)

    for path in args.gates:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        statistics = (TrigramFamiliarity(CACHE, config.vocab_size)
                      if "log_count" in family_features(payload["family"]) else None)
        handle = install_residual_gates(model, payload["layers"],
                                        family=payload["family"],
                                        familiarity=statistics)
        model.residual_gates.load_state_dict(payload["state_dict"], strict=True)
        try:
            gated, stock = [], []
            for _ in range(args.rounds):
                gated.append(measure(model, ids, args.device))
            torch.cuda.reset_peak_memory_stats(args.device)
            measure(model, ids, args.device, repeats=3)
            peak_gated = int(torch.cuda.max_memory_allocated(args.device))
        finally:
            remove_residual_gates(model)
        for _ in range(args.rounds):
            stock.append(measure(model, ids, args.device))
        torch.cuda.reset_peak_memory_stats(args.device)
        measure(model, ids, args.device, repeats=3)
        peak_stock = int(torch.cuda.max_memory_allocated(args.device))

        best_gated, best_stock = min(gated), min(stock)
        report["arms"][payload["family"]] = {
            "gate": str(path),
            "parameters": sum(v.numel() for k, v in payload["state_dict"].items()
                              if k.endswith(("weight", "bias"))),
            "layers": payload["layers"],
            "gated_seconds": best_gated, "stock_seconds": best_stock,
            "overhead_fraction": best_gated / best_stock - 1.0,
            "gated_spread": max(gated) / min(gated) - 1.0,
            "stock_spread": max(stock) / min(stock) - 1.0,
            "peak_gated_bytes": peak_gated, "peak_stock_bytes": peak_stock,
        }
        entry = report["arms"][payload["family"]]
        print("%-12s %+6.2f%% forward  (gated spread %.2f%%, stock spread %.2f%%)  "
              "peak %+d B" % (payload["family"], 100 * entry["overhead_fraction"],
                              100 * entry["gated_spread"], 100 * entry["stock_spread"],
                              peak_gated - peak_stock), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
