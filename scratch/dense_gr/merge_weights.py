"""Blend a fine-tuned checkpoint back toward the one it was trained from, by depth.

merged = base + alpha * (tuned - base), alpha per tensor. Two checkpoints from the same
start a few million tokens apart sit in one basin, so the line between them is a family
of models (WiSE-FT, Wortsman et al. 2022; model averaging also led Lin et al. 2023's
comparison of alignment-tax fixes). LiNeS (Wang et al. 2024) found forgetting lives mostly
in the shallow layers' update and the new skill in the deep layers', so `--shallow` and
`--deep` ramp alpha linearly with depth: layer l gets shallow + (deep - shallow) * l / (L-1),
the embedding (tied to the output) the shallow value, the final norm the deep one.
`--alpha` is the uniform case.

The output is bf16, like its inputs, and many of a short run's updates are a single bf16
step, so a partial alpha rounds each such element to the base or the tuned value: in
effect a pseudo-random subset of the update (as in DARE) rather than an exact fraction of
it. At layer 12 of a 0-to-1 ramp the merged update measures 0.80 of the full one, not 0.52.

    python scratch/dense_gr/merge_weights.py --base <think> --tuned <round> --shallow 0 --deep 1 --output <dir>
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAYER = re.compile(r"\.layers\.(\d+)\.")


def alpha_for(name, layers, shallow, deep):
    match = LAYER.search(name)
    if match:
        return shallow + (deep - shallow) * int(match.group(1)) / (layers - 1)
    return deep if name.startswith("model.norm.") else shallow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tuned", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--shallow", type=float, default=None)
    parser.add_argument("--deep", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.alpha is None) == (args.shallow is None or args.deep is None):
        raise SystemExit("give --alpha, or both --shallow and --deep")
    shallow, deep = (args.alpha, args.alpha) if args.alpha is not None else (args.shallow, args.deep)
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    layers = json.loads((args.base / "config.json").read_text())["num_hidden_layers"]
    merged = {}
    with safe_open(args.base / "model.safetensors", "pt") as base, \
            safe_open(args.tuned / "model.safetensors", "pt") as tuned:
        if set(base.keys()) != set(tuned.keys()):
            raise SystemExit("the checkpoints hold different tensors")
        for name in base.keys():
            a, b = base.get_tensor(name), tuned.get_tensor(name)
            if a.shape != b.shape:
                raise SystemExit("%s differs in shape" % name)
            alpha = alpha_for(name, layers, shallow, deep)
            merged[name] = (a.float() + alpha * (b.float() - a.float())).to(a.dtype)
        metadata = base.metadata()
    args.output.mkdir(parents=True)
    save_file(merged, args.output / "model.safetensors", metadata=metadata)
    for item in args.tuned.iterdir():
        if item.is_file() and item.suffix in (".json", ".jinja") and item.name != "milestone.json":
            shutil.copy2(item, args.output / item.name)
    (args.output / "merge.json").write_text(json.dumps(
        {"base": str(args.base), "tuned": str(args.tuned), "shallow": shallow, "deep": deep},
        indent=1))
    print("merged %d tensors, alpha %.2f (shallow) to %.2f (deep) -> %s"
          % (len(merged), shallow, deep, args.output))


if __name__ == "__main__":
    main()
