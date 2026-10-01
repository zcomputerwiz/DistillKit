"""Int8 weight-only linear layers (W8A16) for the teacher.

bitsandbytes' LLM.int8 quantizes activations as well as weights, and routes the
activation columns holding any value above its threshold through fp16 -- columns chosen
from the whole input. A token's output therefore depends on every other token in the
forward: on the 27B teacher the same positions read with an 8K and a 16K prefix agreed on
the top-1 token only 78% of the time, and on the 2B source (8K input) LLM.int8 sat at KL
2.8e-2 from bf16 against 4.7e-3 for per-channel int8 weights with bf16 activations.

Here weights are symmetric int8 per output channel and activations stay bf16: an int8
value is exact in bf16, so the matmul runs on the integer weights and the scale is
applied to its output. Each position is computed independently of the rest of the
sequence, at the same 1 byte a weight as LLM.int8.
"""
from __future__ import annotations

import torch
from torch import nn


class Int8WeightOnlyLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        weight = linear.weight.data.float()
        scale = weight.abs().amax(1).clamp(min=1e-8) / 127
        self.register_buffer("weight_q", (weight / scale[:, None]).round().clamp(-127, 127).to(torch.int8))
        # fp32: a bf16 scale would add up to 0.4% error to a whole output channel.
        self.register_buffer("scale", scale)
        self.register_buffer("bias", None if linear.bias is None else linear.bias.data.float())
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def forward(self, x):
        out = nn.functional.linear(x, self.weight_q.to(x.dtype)).float() * self.scale
        return (out if self.bias is None else out + self.bias).to(x.dtype)


def quantize_linears(model: nn.Module, skip=("lm_head",)) -> int:
    """Replace every nn.Linear outside `skip` in place; returns how many."""
    count = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and not any(s in full for s in skip):
                setattr(module, child_name, Int8WeightOnlyLinear(child))
                count += 1
    return count
