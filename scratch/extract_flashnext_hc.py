"""Pull Flash-Next's trained hyper-connection routing out of the GGUF.

``WidenedResidual`` is a transcription of this routing, and until now it has been
trained from an identity initialisation at ``n_r=2, lowrank=64`` while the weights it
transcribes sit in the GGUF, trained, at ``n_r=4, lowrank=320``. The shapes line up
exactly once the branch count matches -- 10240 is 4 x 2560 throughout:

    blk.N.hc_attn_down   (10240, 320)   ->  W_down.weight            (320, 10240)
    blk.N.hc_attn_up     (320, 10240)   ->  W_up.weight              (10240, 320)
    blk.N.hc_attn_inject (10240, 4)     ->  W_write.weight           (4, 10240)
    blk.N.hc_attn_norm   (10240,)       ->  branch_gain (4, 2560), see below

Two conventions have to be right, and both are checked against an independent copy
rather than assumed -- ``blk.1.ple_*`` exists in the GGUF *and* in
``../flash-next-ple/ple_layer.pt``, which came from the HF safetensors, so
``--verify`` compares them:

* **Layout.** GGUF reports ``(ne0, ne1)`` with ``ne0`` -- the contracted dimension --
  varying fastest, so the row-major buffer is already ``(out_features, in_features)``:
  exactly what ``nn.Linear`` holds, and reshaping is the entire conversion. Transposing
  would break it, and ``W_down`` at (320, 10240) and its transpose at (10240, 320) are
  both plausible-looking while only one computes anything. Verified: ``ple_key`` and
  ``ple_value`` reshape to the HF tensors with relative error 0.0054, which is Q8_0
  quantisation noise and nothing else.

* **Scale versus deviation.** GGUF stores a norm weight as the scale; HF stores the
  deviation from one, the way ``_PLERMSNorm`` does and for the reason documented
  there. Verified: ``GGUF - 1 == HF`` exactly, to zero, on all three PLE norms. So
  ``hc_attn_norm`` becomes ``branch_gain_delta = scale - 1``, reshaped to per-branch
  rows.

This reads tensor headers and the named tensors only; it never materialises the MoE
experts, which are 1.78 GiB in the same block.

    python scratch/extract_flashnext_hc.py --layers 1 2 3 --output ../flash-next-hc/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

SHARD = ("C:/Users/Owner/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/"
         "snapshots/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ4_XS/"
         "Qwen3.8-Flash-Next-UD-IQ4_XS-%05d-of-00003.gguf")

#: Every block routes its attention and its FFN sublayer separately, which is exactly
#: the ``attn_residual`` / ``mlp_residual`` pair ``WidenedDecoderLayer`` holds. Taking
#: only the attention half would leave the MLP routing at its identity initialisation
#: and call the result "borrowed".
SUBLAYERS = {"hc_attn": "attn_residual", "hc_ffn": "mlp_residual"}

#: GGUF name suffix -> (our attribute, whether it is a matrix rather than a vector)
ROUTING = {
    "norm.weight": ("branch_gain", False),
    "down.weight": ("W_down.weight", True),
    "up.weight": ("W_up.weight", True),
    "inject.weight": ("W_write.weight", True),
}


def dequantize(tensor):
    """Whatever quantisation this tensor carries, as float32."""
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize as gguf_dequantize

    if tensor.tensor_type == GGMLQuantizationType.F32:
        return np.asarray(tensor.data, dtype=np.float32)
    return gguf_dequantize(tensor.data, tensor.tensor_type).astype(np.float32)


def verify_against_hf(reader, reference=Path("../flash-next-ple/ple_layer.pt")):
    """Both conventions, against the HF copy of the same layer. See the module note.

    This is the only independent oracle available for the extraction: every other
    tensor exists in exactly one place, so a silent transpose or an off-by-one in the
    scale convention would go straight into a training run.
    """
    if not reference.exists():
        print("no HF PLE copy at %s; skipping the convention check" % reference)
        return
    held = torch.load(reference, map_location="cpu", weights_only=False)
    found = {t.name: t for t in reader.tensors if t.name.startswith("blk.1.ple_")}
    matrices = {"blk.1.ple_key.weight": "key_proj.weight",
                "blk.1.ple_value.weight": "value_proj.weight"}
    for name, target in matrices.items():
        tensor = found[name]
        shape = tuple(int(dim) for dim in tensor.shape)
        values = torch.from_numpy(dequantize(tensor).copy()).reshape(shape[1], shape[0])
        expected = held[target].float()
        error = float((values - expected).norm() / expected.norm())
        # Q8_0 round trip is about 0.5%; a transpose would be order 1.
        if values.shape != expected.shape or error > 0.02:
            raise ValueError("%s came out %s at relative error %.4f against %s"
                             % (name, tuple(values.shape), error, target))
        print("  layout ok: %-24s relerr %.4f vs %s" % (name, error, target))
    for suffix in ("conv", "key", "query"):
        tensor = found["blk.1.ple_norm_%s.weight" % suffix]
        scale = torch.from_numpy(dequantize(tensor).copy()).flatten()
        expected = held["norm_%s.weight" % suffix].float()
        gap = float((scale - 1.0 - expected).abs().max())
        if gap:
            raise ValueError("norm_%s: GGUF-1 differs from HF by %g; the scale/deviation "
                             "convention is not what this extractor assumes" % (suffix, gap))
        print("  scale ok:  norm_%-5s GGUF - 1 == HF exactly" % suffix)


def find_shard(layer):
    """Which of the three shards holds this block. Shard 1 carries only metadata."""
    from gguf import GGUFReader

    for index in (2, 3):
        reader = GGUFReader(SHARD % index, mode="r")
        if any(t.name.startswith("blk.%d." % layer) for t in reader.tensors):
            return reader, index
    raise KeyError("no shard holds blk.%d" % layer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=2560)
    parser.add_argument("--output", default="../flash-next-hc")
    parser.add_argument("--no-verify", dest="verify", action="store_false",
                        help="skip the layout/scale check against ../flash-next-ple")
    arguments = parser.parse_args()

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    width = arguments.branches * arguments.hidden
    manifest = {"source": SHARD % 2, "branches": arguments.branches,
                "hidden": arguments.hidden, "layers": {}}

    if arguments.verify:
        print("checking the extraction conventions against the HF copy of blk.1.ple:")
        verify_against_hf(find_shard(1)[0])
        print()

    for layer in arguments.layers:
        reader, shard = find_shard(layer)
        found = {t.name.split(".", 2)[2]: t for t in reader.tensors
                 if t.name.startswith("blk.%d." % layer)}
        missing = ["%s_%s" % (prefix, name) for prefix in SUBLAYERS for name in ROUTING
                   if "%s_%s" % (prefix, name) not in found]
        if missing:
            print("blk.%d: no hyper-connection routing (%s); skipping"
                  % (layer, ", ".join(missing)))
            continue
        held = {}
        for prefix, sublayer in SUBLAYERS.items():
            for name, (attribute, is_matrix) in ROUTING.items():
                tensor = found["%s_%s" % (prefix, name)]
                values = torch.from_numpy(dequantize(tensor).copy())
                gguf_shape = tuple(int(s) for s in tensor.shape)
                if is_matrix:
                    # GGUF reports (ne0, ne1) with ne0 -- the contracted dimension --
                    # varying fastest, so the row-major buffer is already
                    # (out_features, in_features): exactly nn.Linear's weight layout.
                    # Reshaping is the whole conversion; transposing would break it.
                    values = values.reshape(gguf_shape[1], gguf_shape[0]).contiguous()
                else:
                    values = values.reshape(-1)
                if attribute == "branch_gain":
                    if values.numel() != width:
                        raise ValueError("%s_norm is %d, expected %d"
                                         % (prefix, values.numel(), width))
                    # WidenedResidual stores the deviation from unit gain, not the gain.
                    held["%s.branch_gain_delta" % sublayer] = (
                        values.view(arguments.branches, arguments.hidden) - 1.0).contiguous()
                else:
                    held["%s.%s" % (sublayer, attribute)] = values
        for sublayer in SUBLAYERS.values():
            lowrank = held["%s.W_down.weight" % sublayer].shape[0]
            expected = {
                "W_down.weight": (lowrank, width),
                "W_up.weight": (width, lowrank),
                "W_write.weight": (arguments.branches, width),
                "branch_gain_delta": (arguments.branches, arguments.hidden),
            }
            for attribute, want in expected.items():
                actual = tuple(held["%s.%s" % (sublayer, attribute)].shape)
                if actual != want:
                    raise ValueError("%s.%s came out %s, expected %s -- check the layout"
                                     % (sublayer, attribute, actual, want))
        path = output / ("layer-%02d.pt" % layer)
        torch.save(held, path)
        lowranks = {name: int(held["%s.W_down.weight" % name].shape[0])
                    for name in SUBLAYERS.values()}
        manifest["layers"][str(layer)] = {
            "shard": shard, "file": path.name, "lowrank": lowranks,
            "shapes": {key: list(value.shape) for key, value in held.items()},
            "norms": {key: float(value.float().norm()) for key, value in held.items()},
        }
        print("blk.%d -> %s  %d tensors, lowrank %s"
              % (layer, path.name, len(held), lowranks))

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nwrote", output / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
