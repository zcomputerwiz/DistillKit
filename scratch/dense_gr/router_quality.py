"""How much of the teacher's attention the router's blocks actually capture.

`selected` cannot answer this. It is `chosen / eligible` with a fixed top-k, so it is a
constant of the geometry -- two routers with completely different choices report the same
number, and reading equality between a trained and a random router as "the choices are no
more diverse" was wrong.

The measure that means something is mass. Take the source model's dense attention, pool it
to blocks, and ask what share of it the student's selected blocks cover, against two
references:

* **recency** -- open the k most recent eligible blocks and nothing else. Attention is
  recency-heavy, so any router has to beat this to have earned its parameters.
* **oracle** -- open the k highest-mass eligible blocks. The ceiling for any router at
  this k, and the gap to it is what is left to win.

Reported on held-out windows the router never trained on.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import triton_shim  # noqa: F401,E402

from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402
from distillkit.models.qwen35.csa2 import Qwen35SparseLatentAttention  # noqa: E402

STORE = Path("scratch/code_training/tokens-v2")


def held_out(store, vocab, count, length, device):
    """The split the arms score on, the windows they score on."""
    stream = np.memmap(store / ("calibration-v%d.bin" % vocab), dtype=np.uint16, mode="r")
    starts = np.random.default_rng(12345).integers(0, stream.shape[0] - length - 1,
                                                   size=count)
    for start in starts:
        yield torch.from_numpy(
            np.array(stream[start:start + length], dtype=np.int64).reshape(1, length)
        ).to(device)


def teacher_mass(model, ids, block):
    """Block-pooled attention of the source, per full-attention layer."""
    caught = {}

    def catcher(index):
        def hook(module, args, kwargs, output):
            caught[index] = output[1][0].float().mean(0)
        return hook

    full = [i for i, kind in enumerate(model.config.layer_types)
            if "linear" not in str(kind)]
    handles = [model.model.layers[i].self_attn.register_forward_hook(
        catcher(i), with_kwargs=True) for i in full]
    with torch.no_grad():
        model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    for handle in handles:
        handle.remove()
    blocks = ids.shape[1] // block
    return {i: w.reshape(blocks, block, blocks, block).sum((1, 3))
            for i, w in caught.items()}


def student_choice(model, ids):
    """`last_allowed` for every routing layer, after one forward."""
    with torch.no_grad():
        model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    return {i: layer.self_attn.last_allowed[0]
            for i, layer in enumerate(model.model.layers)
            if isinstance(getattr(layer, "self_attn", None), Qwen35SparseLatentAttention)
            and layer.self_attn.last_allowed is not None}


def captured(mass, chosen, eligible):
    """Share of the eligible attention mass that ``chosen`` covers, per query block."""
    total = (mass * eligible).sum(-1)
    rows = total > 1e-9
    if not rows.any():
        return None
    return float(((mass * (chosen & eligible)).sum(-1)[rows] / total[rows]).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--student", type=Path, nargs="+", required=True)
    parser.add_argument("--windows", type=int, default=32)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    source = Qwen35WidenedForCausalLM.from_pretrained(
        args.source, dtype=torch.float32, attn_implementation="eager").to(device).eval()

    report = {}
    for path in args.student:
        student = Qwen35WidenedForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16).to(device).eval()
        block = student.config.csa2_block_size
        vocab = student.config.vocab_size
        rows = {}
        for ids in held_out(args.store, vocab, args.windows, args.length, device):
            mass = teacher_mass(source, ids, block)
            chosen = student_choice(student, ids)
            blocks = ids.shape[1] // block
            index = torch.arange(blocks, device=device)
            offsets = index.view(-1, 1) - index.view(1, -1)
            for layer, allowed in chosen.items():
                if layer not in mass:
                    continue
                attention = mass[layer]
                eligible = offsets > student.model.layers[layer].self_attn.local_blocks
                keep = int((allowed & eligible).sum(-1).max())
                recent = torch.zeros_like(allowed)
                if keep:
                    order = offsets.masked_fill(~eligible, 1 << 20).argsort(-1)[:, :keep]
                    recent.scatter_(-1, order, True)
                oracle = torch.zeros_like(allowed)
                if keep:
                    best = attention.masked_fill(~eligible, -1.0).topk(keep, -1).indices
                    oracle.scatter_(-1, best, True)
                slot = rows.setdefault(layer, {"router": [], "recency": [], "oracle": []})
                for name, pick in (("router", allowed), ("recency", recent),
                                   ("oracle", oracle)):
                    value = captured(attention, pick, eligible)
                    if value is not None:
                        slot[name].append(value)
        report[str(path)] = {layer: {k: float(np.mean(v)) for k, v in slot.items()}
                             for layer, slot in sorted(rows.items())}
        del student
        torch.cuda.empty_cache()

    print("share of the teacher's eligible attention mass the selected blocks capture")
    print("%-34s %-6s %8s %8s %8s" % ("checkpoint", "layer", "router", "recency", "oracle"))
    for name, layers in report.items():
        for layer, slot in layers.items():
            print("%-34s %-6d %8.4f %8.4f %8.4f"
                  % (Path(name).name, layer, slot["router"], slot["recency"],
                     slot["oracle"]))
    if args.output:
        args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print("\nwrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
