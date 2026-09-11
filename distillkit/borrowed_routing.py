"""Initialise the widened residual from Flash-Next's trained hyper-connection routing.

``WidenedResidual`` is a transcription of the ``hc_attn_*`` / ``hc_ffn_*`` tensors that
every Flash-Next block carries. Until now it has trained from an identity
initialisation while the weights it transcribes sat unused in the GGUF, because at
``n_r=2`` none of them fit -- every one is sized ``4 * 2560``.

**This is initialisation, not transplant.** The routing trains from here like any other
parameter, and the only claim being made is that Flash-Next's trained values are a
better starting point than identity. Whether they are is what the run measures.

**The lambdas have to be turned on, or the borrow is inert.** ``_combine`` computes
``correction = read_offset + lambda_read * read_gate`` and
``weights = (1 + write_offset) + lambda_write * sigmoid(write_logits)``, and both
lambdas initialise to zero so the identity route survives a fresh widening. That means
``W_down``, ``W_up`` and ``W_write`` -- the three tensors being borrowed -- are
multiplied by zero. Copying trained values into them and leaving the lambdas alone
changes nothing in the forward: the run trains, reports numbers, and the borrow did not
happen. ``branch_gain_delta`` is the trap's sharp edge, because it rides on the norm
and *does* take effect, so half the transfer would work and half would not.

Flash-Next has no equivalent of these lambdas -- its ``hc_*`` tensors are the routing,
not a deviation from an identity route -- so borrowing sets them to one, which is the
closest reading of "the donor's gate, fully on". That deliberately gives up identity at
load: a borrowed model does not reproduce the un-widened student at step zero, and it
is not supposed to. ``bypassed`` still means what it meant, since it removes the
sidecar and leaves the widening.

**The layer correspondence is the unresolved part.** Flash-Next has 48 blocks to this
student's 32, and the depth sweep (``PROGRESS.md``, 2026-09-10) found this student wants
the sidecar at layer 24 of 32 while Flash-Next puts its PLE at block 1 of 48. Position
in the stack demonstrably does not transfer, so neither map here is known to be right
and ``init_layer_map`` exists to test them rather than to encode an answer.

The extraction and its two conventions -- GGUF layout, and scale-versus-deviation --
are validated in ``scratch/extract_flashnext_hc.py`` against an independent HF copy of
``blk.1.ple``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch import nn

LOG = logging.getLogger(__name__)

#: One file per donor block, holding both sublayers' routing.
SUBLAYERS = ("attn_residual", "mlp_residual")
TENSORS = ("W_down.weight", "W_up.weight", "W_write.weight", "branch_gain_delta")


def layer_map(our_layers: int, donor_blocks: list[int], how: str) -> dict[int, int]:
    """Which donor block initialises each of our layers.

    ``proportional`` spreads our stack over the donor's, ``identity`` uses the donor's
    bottom ``our_layers`` blocks. Both are guesses; see the module note.
    """
    if how == "identity":
        chosen = [block for block in donor_blocks if block < our_layers]
        if len(chosen) < our_layers:
            raise ValueError(
                f"identity map needs donor blocks 0..{our_layers - 1}, found {len(chosen)}")
        return {layer: layer for layer in range(our_layers)}
    if how != "proportional":
        raise ValueError(f"unknown init_layer_map: {how!r}")
    available = sorted(donor_blocks)
    span = len(available)
    # Spread across the donor's full depth, so our last layer lands on its last block.
    return {layer: available[min(span - 1, round(layer * (span - 1) / max(1, our_layers - 1)))]
            for layer in range(our_layers)}


def load_manifest(directory: Path) -> dict:
    manifest = directory / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} not found; run scratch/extract_flashnext_hc.py first")
    return json.loads(manifest.read_text(encoding="utf-8"))


def initialise_widened_residual(model, directory, how="proportional") -> dict:
    """Copy donor routing into every ``WidenedResidual``. Returns what it did.

    Shapes are checked rather than reshaped. A mismatch here means the run's
    ``num_branches`` or ``lowrank`` disagrees with the extraction, and silently
    adapting it would train something other than what the config asked for.
    """
    directory = Path(directory)
    manifest = load_manifest(directory)
    blocks = manifest["blocks"]
    layers = model.model.layers
    mapping = layer_map(len(layers), blocks, how)

    cache: dict[int, dict] = {}
    copied, skipped = 0, []
    for index, layer in enumerate(layers):
        block = mapping[index]
        if block not in cache:
            cache[block] = torch.load(directory / ("layer-%02d.pt" % block),
                                      map_location="cpu", weights_only=False)
        donor = cache[block]
        for sublayer in SUBLAYERS:
            module = getattr(layer, sublayer, None)
            if module is None:
                skipped.append(f"{index}.{sublayer}")
                continue
            for name in TENSORS:
                source = donor["%s.%s" % (sublayer, name)]
                target = module.get_parameter(name)
                if tuple(target.shape) != tuple(source.shape):
                    raise ValueError(
                        f"layer {index}.{sublayer}.{name} is {tuple(target.shape)} but the "
                        f"donor is {tuple(source.shape)}; residual_stream.num_branches and "
                        f"lowrank must match the extraction "
                        f"({manifest['branches']} branches, {manifest['hidden']} hidden)")
                with torch.no_grad():
                    target.copy_(source.to(device=target.device, dtype=target.dtype))
                copied += 1
            # Without this the three borrowed matrices are multiplied by zero and the
            # transfer is inert. See the module note.
            with torch.no_grad():
                module.lambda_read.fill_(1.0)
                module.lambda_write.fill_(1.0)

    report = {"directory": str(directory), "map": how, "layers": len(layers),
              "donor_blocks": len(blocks), "tensors_copied": copied,
              "sample_mapping": {str(k): mapping[k] for k in sorted(mapping)[:4]},
              "sidecar_layer_donor": mapping.get(
                  getattr(model.config, "sidecar_layer_index", -1))}
    if skipped:
        report["skipped"] = skipped
    LOG.info("Borrowed routing: %d tensors over %d layers from %d donor blocks (%s map); "
             "layer %s <- block %s", copied, len(layers), len(blocks), how,
             getattr(model.config, "sidecar_layer_index", "?"),
             report["sidecar_layer_donor"])
    return report


def initialise_ple_reader(model, reference) -> dict:
    """Flash-Next's trained PLE into the sidecar, where the widths allow it.

    ``key_proj`` is ``(4 * 2560, 2560)`` -- one key per branch -- and the three norms
    and the convolution are all ``4 * 2560`` wide, so this only applies to the ``ple``
    variant at four branches. ``value_proj`` is ``(2560, 2560)`` and is the one tensor
    that fits at any width; it is copied whenever the variant has one.
    """
    reference = Path(reference)
    if not reference.exists():
        raise FileNotFoundError(f"{reference} not found")
    held = torch.load(reference, map_location="cpu", weights_only=False)
    sidecar = model.model.layers[model.config.sidecar_layer_index].sidecar
    module = getattr(sidecar, "ple", sidecar)
    names = {"key_proj.weight": "key_proj.weight",
             "value_proj.weight": "value_proj.weight",
             "norm_key.weight": "norm_key.weight",
             "norm_query.weight": "norm_query.weight",
             "norm_conv.weight": "norm_conv.weight",
             "conv1d.weight": "conv1d.weight"}
    copied, mismatched = [], []
    for ours, theirs in names.items():
        if theirs not in held:
            continue
        try:
            target = module.get_parameter(ours)
        except AttributeError:
            continue
        source = held[theirs]
        if theirs == "conv1d.weight" and source.ndim == 2:
            source = source.unsqueeze(1)
        if tuple(target.shape) != tuple(source.shape):
            mismatched.append(f"{ours}: ours {tuple(target.shape)} donor {tuple(source.shape)}")
            continue
        with torch.no_grad():
            target.copy_(source.to(device=target.device, dtype=target.dtype))
        copied.append(ours)
    report = {"reference": str(reference), "copied": copied, "mismatched": mismatched}
    LOG.info("Borrowed PLE reader: copied %s%s", copied or "nothing",
             ("; width mismatch on " + ", ".join(mismatched)) if mismatched else "")
    return report
