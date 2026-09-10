"""Three claims about restructuring the PLE layer, checked rather than argued.

The gate decomposition said the key path -- which is 80% of the layer's multiplies --
produces four scalars that barely depend on the row it is given. That invites a rewrite,
but two of the available savings are exact algebra rather than approximation, and those
are worth separating from the one that is a measured approximation.

1. **The convolution branch cannot see the gate.** `norm_conv` RMS-normalises
   `gate * value`, and RMS normalisation cancels a positive scalar, so the branch's input
   is `norm_conv(value)` regardless of the gate. Equal up to `eps`, which only matters
   when `gate^2 * mean(value^2)` is not comfortably above 1e-6.

2. **`norm_conv`'s weight folds into `conv1d`.** The convolution is depthwise, so scaling
   channel c of its input by a_c is the same as scaling that channel's filter. The whole
   parameter can be folded at load time.

3. Together those mean the `[batch, seq, hc*hidden]` gated tensor never has to exist for
   the convolution branch, and one reduction serves all `hc` streams instead of `hc`
   reductions over identical data.

    python scratch/ple_restructure_check.py
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def rms_norm(x, weight, streams, eps=1e-6):
    grouped = x.unflatten(-1, (streams, x.shape[-1] // streams)).float()
    grouped = grouped * torch.rsqrt(grouped.pow(2).mean(-1, keepdim=True) + eps)
    return (grouped.flatten(-2) * (1.0 + weight.float())).to(x.dtype)


def report(name, reference, candidate):
    scale = reference.abs().max().clamp_min(1e-12)
    relative = ((candidate - reference).abs().max() / scale).item()
    print("%-58s max relative difference %.3e" % (name, relative))
    return relative


def main():
    torch.manual_seed(0)
    batch, sequence, hidden, streams = 2, 128, 2560, 4
    eps = 1e-6

    value = torch.randn(batch, sequence, hidden) * 1.5
    gate = torch.rand(batch, sequence, streams, 1) * 0.9 + 0.05
    conv_weight = torch.randn(streams * hidden) * 0.1

    gated = gate * value.unsqueeze(-2)
    reference = rms_norm(gated.flatten(-2), conv_weight, streams, eps)

    # 1: the same thing computed from `value` alone, with no gate anywhere.
    shared = rms_norm(value.repeat(1, 1, streams), conv_weight, streams, eps)
    first = report("norm_conv(gate * value) vs norm_conv(value)", reference, shared)

    # ... and the eps sensitivity that bounds it: shrink the gate and the agreement
    # degrades exactly as eps/(gate^2 * mean(value^2)) predicts.
    for size in (1.0, 1e-1, 1e-2, 1e-3):
        small = torch.full((batch, sequence, streams, 1), size)
        left = rms_norm((small * value.unsqueeze(-2)).flatten(-2), conv_weight, streams, eps)
        predicted = eps / (size ** 2 * value.pow(2).mean()).item()
        print("    gate = %-6g  relative difference %.3e   eps/(gate^2 mean(v^2)) = %.3e"
              % (size, ((left - shared).abs().max() / shared.abs().max()).item(), predicted))

    # 2: a depthwise convolution absorbs a per-channel input scaling into its filters.
    conv = nn.Conv1d(streams * hidden, streams * hidden, kernel_size=4,
                     groups=streams * hidden, dilation=3, bias=False)
    plain = rms_norm(value.repeat(1, 1, streams), torch.zeros(streams * hidden), streams, eps)
    padded = F.pad(shared.transpose(1, 2), (9, 0))
    reference_conv = F.silu(conv(padded)).transpose(1, 2)
    folded = nn.Conv1d(streams * hidden, streams * hidden, kernel_size=4,
                       groups=streams * hidden, dilation=3, bias=False)
    with torch.no_grad():
        folded.weight.copy_(conv.weight * (1.0 + conv_weight).view(-1, 1, 1))
    candidate_conv = F.silu(folded(F.pad(plain.transpose(1, 2), (9, 0)))).transpose(1, 2)
    second = report("silu(conv(norm_conv(x))) vs silu(folded_conv(rms(x)))",
                    reference_conv, candidate_conv)

    # 3: one reduction for every stream, since the streams differ only by a diagonal.
    once = value.float() * torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + eps)
    expanded = (once.repeat(1, 1, streams) * (1.0 + conv_weight)).to(value.dtype)
    third = report("hc grouped reductions vs one reduction plus a diagonal",
                   shared, expanded)

    print()
    key_macs = hidden * streams * hidden
    value_macs = hidden * hidden
    gate_macs = streams * hidden
    conv_macs = streams * hidden * 4
    total = key_macs + value_macs + gate_macs + conv_macs
    print("multiplies per token at hidden=%d, hc=%d:" % (hidden, streams))
    for name, count in (("key_proj", key_macs), ("value_proj", value_macs),
                        ("gate dot", gate_macs), ("depthwise conv", conv_macs)):
        print("    %-16s %10d  %5.1f%%" % (name, count, 100 * count / total))
    print("    %-16s %10d" % ("total", total))
    print("    key_proj replaced by a learned vector per stream: %d multiplies, %.0fx less"
          % (gate_macs, key_macs / gate_macs))

    # The first claim is exact only as eps -> 0, so it is checked against the bound the
    # sweep above confirms rather than against a threshold picked to pass.
    bound = eps / (gate.min().item() ** 2 * value.pow(2).mean().item())
    assert first < bound, (first, bound)
    assert second < 1e-5, second
    assert third == 0.0, third
    print("claim 1 bound at the smallest gate in this sample: %.3e" % bound)
    print("\nall three hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
