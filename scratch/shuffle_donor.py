"""A donor whose weights carry no donor information, for the control arm.

The blend sweep found that a small amount of donor routing improves assistant NLL
before any training: 0.5039 at blend 0.10 against 0.5176 at 0. That is either the
borrowed routing contributing something, or it is the interpolation happening to
rescale the residual in a way this student likes. Those are very different findings and
the numbers alone cannot tell them apart.

So: the same tensors with their elements randomly permuted. Every tensor keeps its
shape and its exact multiset of values, so the scale, the distribution, the sparsity and
the norm are all preserved to the last bit -- and the structure that made them a trained
routing is gone. If the control moves NLL the same way, the gain is rescaling and there
is no borrowing happening.

Permuting within each tensor is the stronger control. Permuting *which donor block goes
to which layer* would leave each tensor internally intact and only break the depth
correspondence, which is a weaker claim and one the proportional map has not earned
anyway.

    python scratch/shuffle_donor.py --source ../flash-next-hc --output ../flash-next-hc-shuffled
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="../flash-next-hc")
    parser.add_argument("--output", default="../flash-next-hc-shuffled")
    parser.add_argument("--seed", type=int, default=20260911)
    arguments = parser.parse_args()

    source, output = Path(arguments.source), Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    generator = torch.Generator().manual_seed(arguments.seed)

    checked = 0
    for block in manifest["blocks"]:
        name = "layer-%02d.pt" % block
        held = torch.load(source / name, map_location="cpu", weights_only=False)
        shuffled = {}
        for key, tensor in held.items():
            flat = tensor.reshape(-1)
            shuffled[key] = flat[torch.randperm(flat.numel(), generator=generator)].reshape(
                tensor.shape).contiguous()
            # The whole point of the control is that only the arrangement changed.
            if not torch.equal(torch.sort(shuffled[key].reshape(-1)).values,
                               torch.sort(flat).values):
                raise ValueError("%s/%s: the value multiset changed" % (name, key))
            checked += 1
        torch.save(shuffled, output / name)

    manifest["shuffled_from"] = str(source.resolve())
    manifest["shuffle_seed"] = arguments.seed
    manifest["shuffle"] = "uniform random permutation of elements within each tensor"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("shuffled %d tensors over %d blocks -> %s"
          % (checked, len(manifest["blocks"]), output))
    print("every tensor keeps its shape and its exact multiset of values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
