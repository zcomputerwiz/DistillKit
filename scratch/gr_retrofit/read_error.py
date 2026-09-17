"""Where the asymmetric mode's NLL change comes from, measured per sublayer.

The conversion gate reports an end-to-end number. It cannot say whether that number is
an implementation defect or the construction behaving as derived, and the difference
decides whether there is anything to fix.

Each sublayer's read is compared against the norm it replaced, on the real recipient in
the deployed precision, with every branch still holding the same state. Symmetric should
be exactly zero at every sublayer: `2 * gamma` differs from `gamma` by an exact power of
two, so the branch normalization rounds to the same significand and the gate of exactly
one half undoes the scale. Asymmetric cannot be, because four perturbed gains average to
`2 * gamma` only after summation, and each branch is rounded before that sum.

    CUDA_VISIBLE_DEVICES=0 python scratch/gr_retrofit/read_error.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from distillkit.experimental import HyperConnection

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_gate import documents, load_converted, load_original  # noqa: E402


@torch.inference_mode()
def measure(model, ids, device):
    """Max |read - norm(branch)| at every sublayer of one forward."""
    errors, handles = [], []

    def watch(module):
        original = module.read

        def patched(states, norm):
            read, weights = original(states, norm)
            reference = norm(states[..., module.read_index, :])
            errors.append({"read": float((read.float() - reference.float()).abs().max()),
                           "write": float((weights.float() - 1.0).abs().max()),
                           "branch_spread": float(
                               (states - states[..., :1, :]).float().abs().max())})
            return read, weights

        module.read = patched
        handles.append((module, original))

    for module in model.modules():
        if isinstance(module, HyperConnection):
            watch(module)
    tokens = torch.tensor([ids], device=device)
    model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False)
    for module, original in handles:
        module.read = original
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path,
                        default=Path("scratch/gr_retrofit/read-error.json"))
    args = parser.parse_args()

    device, dtype = args.device, torch.bfloat16
    record = documents(1)[0]
    original, config = load_original(device, dtype)
    del original
    if args.device == "cuda":
        torch.cuda.empty_cache()

    report = {"document": record["id"], "dtype": str(dtype), "modes": {}}
    for mode in ("symmetric", "asymmetric"):
        model, _ = load_converted(config, device, dtype, mode, 0, 4, 320)
        rows = measure(model, record["ids"], device)
        report["modes"][mode] = {
            "sublayers": len(rows),
            "max_read_error": max(row["read"] for row in rows),
            "first_sublayer_read_error": rows[0]["read"],
            "exact_sublayers": sum(1 for row in rows if row["read"] == 0.0),
            "max_write_error": max(row["write"] for row in rows),
            "max_branch_spread": max(row["branch_spread"] for row in rows),
            "per_sublayer_read_error": [row["read"] for row in rows]}
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for mode, rows in report["modes"].items():
        print("%-11s exact %d/%d sublayers, max read error %.3e, first %.3e, "
              "write error %.3e, branch spread %.3e"
              % (mode, rows["exact_sublayers"], rows["sublayers"],
                 rows["max_read_error"], rows["first_sublayer_read_error"],
                 rows["max_write_error"], rows["max_branch_spread"]))
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
