"""Where the backbones moved, and whether the gate changed where they went.

Two questions the NLL numbers cannot answer. First, does carrying a warm-started gate send
the backbone somewhere materially different, or does it move the same way and simply end
up paired with a different routing policy? Second, of the movement that does differ
between arms, how much is larger than the movement that differs between two runs of the
*same* arm -- which the repeat design showed is not negligible.

Both are questions about ``delta B = B_endpoint - B_0`` per module, and both are answered
on CPU from safetensors without loading a model.

    python scratch/residual_gate/backbone_delta.py --output scratch/residual_gate/forensics/delta.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

RUNS = Path("D:/DeepThought/Projects/HybridModel/runs")
BASE = Path("D:/DeepThought/Projects/HybridModel/student-2b-hf")
LAYER = re.compile(r"^model\.layers\.(\d+)\.")
GATED = (10, 12, 14, 16)

ARMS = {
    "A42": "gate-coadapt-armA", "A42r1": "gate-coadapt-armA-r1",
    "A42r2": "gate-coadapt-armA-r2", "B42": "gate-coadapt-armB",
    "C42": "gate-coadapt-armC", "C42r1": "gate-coadapt-armC-r1",
    "C42r2": "gate-coadapt-armC-r2",
    "A43": "gate-coadapt-s43-armA", "A43m": "gate-coadapt-s43m-armA",
    "A43r": "gate-coadapt-s43r-armA", "B43": "gate-coadapt-s43-armB",
    "C43": "gate-coadapt-s43-armC", "C43r1": "gate-coadapt-s43-armC-r1",
    "C43r2": "gate-coadapt-s43-armC-r2",
    "D42": "gate-d-42-r0", "D42r1": "gate-d-42-r1", "D42r2": "gate-d-42-r2",
    "D43": "gate-d-43-r0", "D43r1": "gate-d-43-r1", "D43r2": "gate-d-43-r2",
    "B42r1": "gate-coadapt-armB-r1", "B42r2": "gate-coadapt-armB-r2",
    "B43m": "gate-coadapt-s43m-armB", "B43r": "gate-coadapt-s43r-armB",
}


def submodule(name: str) -> str:
    """Coarse enough to compare, fine enough to locate: attention, MLP, norms, rest."""
    if ".mlp." in name:
        return "mlp"
    if ".self_attn." in name or ".linear_attn." in name:
        return "attention"
    if "layernorm" in name or name.endswith(".norm.weight"):
        return "norm"
    if "embed_tokens" in name or "lm_head" in name:
        return "embedding"
    return "other"


def shards(path: Path):
    index = path / "model.safetensors.index.json"
    if index.exists():
        files = sorted({name for name in
                        json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()})
        return [path / name for name in files]
    single = path / "model.safetensors"
    if not single.exists():
        raise SystemExit("no safetensors under %s" % path)
    return [single]


def deltas(checkpoint: Path, base: dict) -> dict:
    """Per (layer, submodule) squared movement and the flattened delta for cosines."""
    summary = {}
    vectors = {}
    for shard in shards(checkpoint):
        with safe_open(str(shard), framework="pt") as handle:
            for name in handle.keys():
                if name.startswith("residual_gates."):
                    continue
                reference = base.get(name)
                if reference is None:
                    continue
                value = handle.get_tensor(name).float()
                difference = (value - reference).reshape(-1)
                match = LAYER.match(name)
                key = ("layer%s" % match.group(1) if match else "outside",
                       submodule(name))
                entry = summary.setdefault(key, {"moved": 0.0, "base": 0.0, "count": 0})
                entry["moved"] += float((difference ** 2).sum())
                entry["base"] += float((reference.reshape(-1) ** 2).sum())
                entry["count"] += difference.numel()
                vectors.setdefault(key, []).append(difference)
    out = {}
    for key, entry in summary.items():
        label = "%s/%s" % key
        out[label] = {
            "l2": entry["moved"] ** 0.5,
            "rms": (entry["moved"] / max(entry["count"], 1)) ** 0.5,
            "relative": (entry["moved"] ** 0.5) / (entry["base"] ** 0.5 + 1e-12),
            "parameters": entry["count"],
        }
    flat = {"%s/%s" % key: torch.cat(parts) for key, parts in vectors.items()}
    return out, flat


def cosine(left, right) -> float:
    return float(torch.nn.functional.cosine_similarity(
        left.unsqueeze(0), right.unsqueeze(0)).item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, default=284)
    parser.add_argument("--arms", nargs="+",
                        default=["A42", "C42", "B42", "A43", "C43", "B43",
                                 "A42r1", "C42r1", "A43r", "C43r1"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    base = {}
    for shard in shards(BASE):
        with safe_open(str(shard), framework="pt") as handle:
            for name in handle.keys():
                base[name] = handle.get_tensor(name).float()
    print("base: %d tensors" % len(base), flush=True)

    summaries, flats = {}, {}
    for name in args.arms:
        checkpoint = RUNS / ARMS[name] / ("checkpoint-%d" % args.step)
        if not checkpoint.is_dir():
            print("skipping %s: no checkpoint-%d" % (name, args.step))
            continue
        summaries[name], flats[name] = deltas(checkpoint, base)
        gated = [summaries[name]["layer%d/mlp" % index]["relative"] for index in GATED
                 if "layer%d/mlp" % index in summaries[name]]
        print("%-6s gated-layer MLP relative movement %s"
              % (name, ["%.5f" % value for value in gated]), flush=True)

    report = {"base": str(BASE), "step": args.step, "modules": summaries,
              "cosine": {}, "norm_ratio": {}}
    names = list(flats)
    for index, left in enumerate(names):
        for right in names[:index]:
            keys = sorted(set(flats[left]) & set(flats[right]))
            report["cosine"]["%s vs %s" % (left, right)] = {
                key: cosine(flats[left][key], flats[right][key]) for key in keys}
            report["norm_ratio"]["%s vs %s" % (left, right)] = {
                key: (float(flats[left][key].norm())
                      / (float(flats[right][key].norm()) + 1e-12)) for key in keys}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
