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
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAYER = re.compile(r"\.layers\.(\d+)\.")


def alpha_for(name, layers, shallow, deep):
    match = LAYER.search(name)
    if match:
        return shallow + (deep - shallow) * int(match.group(1)) / (layers - 1)
    return deep if name.startswith("model.norm.") else shallow


def qk_rows(name, shape, config):
    """The part of `name` that is a softmax-attention layer's query or key map: True for
    the whole tensor, a row mask for part of it, None for none of it.

    QK-Restore (Attention Amnesia, arXiv 2606.11052): fine-tuning a distilled hybrid on
    reasoning traces bent its few softmax-attention layers' W_Q and W_K toward short-range
    routing (needle-in-a-haystack 67.2% -> 9.4% on HypeNet-9B), and taking those two back
    from the checkpoint before the fine-tune restored long-range recall while keeping
    nearly all it taught (MATH500 -0.6). Only the attention layers; recall in a distilled
    hybrid rides on them. Here a head's query is the first half of its `q_proj` rows (the
    second half is the output gate) and its `q_norm`; its key is the rotary rows of
    `kv_a_proj` and its content-key rows of `kv_b_proj`, read from the latent. The latent
    also feeds the values, so it stays tuned, as the residual stream feeding W_K does in
    the paper. The CSA2 indexer only routes, so all of it.
    """
    from distillkit.models.qwen35.csa2 import ROUTER_PARAMETER_PARTS
    from distillkit.models.qwen35.mla import mla_dimensions

    if ".self_attn." not in name:
        return None
    module = name.split(".self_attn.", 1)[1].rsplit(".", 1)[0]
    if module in ROUTER_PARAMETER_PARTS or module == "q_norm":
        return True
    latent, rope_dim, content = mla_dimensions(config)
    head = config.head_dim
    widths = {"q_proj": 2 * head, "kv_a_proj": latent + rope_dim, "kv_b_proj": content + head}
    if module not in widths:
        return None
    if shape[0] % widths[module]:
        raise SystemExit("%s: %d rows is not a whole number of %d-row heads" % (name, shape[0], widths[module]))
    row = torch.arange(shape[0]) % widths[module]
    return {"q_proj": row < head, "kv_a_proj": row >= latent, "kv_b_proj": row < content}[module]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tuned", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--shallow", type=float, default=None)
    parser.add_argument("--deep", type=float, default=None)
    parser.add_argument("--also", nargs="*", default=[], metavar="PATH:SHALLOW:DEEP",
                        help="more tuned checkpoints from the same base, each update added with its "
                             "own depth ramp (task arithmetic): base + sum of alpha_i x update_i")
    parser.add_argument("--restore-qk", action="store_true",
                        help="then take the attention layers' query and key maps from --base (qk_rows)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.alpha is None) == (args.shallow is None or args.deep is None):
        raise SystemExit("give --alpha, or both --shallow and --deep")
    shallow, deep = (args.alpha, args.alpha) if args.alpha is not None else (args.shallow, args.deep)
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    config = SimpleNamespace(**json.loads((args.base / "config.json").read_text()))
    layers = config.num_hidden_layers
    restored = 0
    merged = {}
    extra = [(Path(p), float(s), float(d)) for p, s, d in (spec.rsplit(":", 2) for spec in args.also)]
    with safe_open(args.base / "model.safetensors", "pt") as base, \
            safe_open(args.tuned / "model.safetensors", "pt") as tuned:
        others = [(safe_open(p / "model.safetensors", "pt"), s, d) for p, s, d in extra]
        if any(set(base.keys()) != set(h.keys()) for h in [tuned] + [h for h, _, _ in others]):
            raise SystemExit("the checkpoints hold different tensors")
        for name in base.keys():
            a = base.get_tensor(name).float()
            total = a.clone()
            for handle, s, d in [(tuned, shallow, deep)] + others:
                b = handle.get_tensor(name).float()
                if a.shape != b.shape:
                    raise SystemExit("%s differs in shape" % name)
                total += alpha_for(name, layers, s, d) * (b - a)
            rows = qk_rows(name, a.shape, config) if args.restore_qk else None
            if rows is not None:
                if rows is True:
                    total = a.clone()
                else:
                    total[rows] = a[rows]
                restored += 1
            merged[name] = total.to(base.get_tensor(name).dtype)
        metadata = base.metadata()
    args.output.mkdir(parents=True)
    save_file(merged, args.output / "model.safetensors", metadata=metadata)
    for item in args.tuned.iterdir():
        if item.is_file() and item.suffix in (".json", ".jinja") and item.name != "milestone.json":
            shutil.copy2(item, args.output / item.name)
    (args.output / "merge.json").write_text(json.dumps(
        {"base": str(args.base), "tuned": str(args.tuned), "shallow": shallow, "deep": deep,
         "also": [[str(p), s, d] for p, s, d in extra], "restore_qk": args.restore_qk},
        indent=1))
    print("merged %d tensors, alpha %.2f (shallow) to %.2f (deep)%s -> %s"
          % (len(merged), shallow, deep,
             ", query/key of %d from the base" % restored if args.restore_qk else "", args.output))


if __name__ == "__main__":
    main()
