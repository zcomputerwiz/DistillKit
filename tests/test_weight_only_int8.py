"""Int8 weight-only linears: per-channel rounding error only, and row-independent."""
import torch
from torch import nn

from distillkit.weight_only_int8 import Int8WeightOnlyLinear, quantize_linears


def test_matches_dequantized_weights_and_ignores_other_rows():
    torch.manual_seed(0)
    linear = nn.Linear(64, 32, bias=True).to(torch.bfloat16)
    q = Int8WeightOnlyLinear(linear)
    x = torch.randn(5, 64, dtype=torch.bfloat16)
    dequantized = q.weight_q.float() * q.scale.float()[:, None]
    torch.testing.assert_close(q(x).float(), x.float() @ dequantized.T + linear.bias.float(), atol=0.05, rtol=0.02)
    # A row's output does not depend on the rows beside it.
    torch.testing.assert_close(q(x[:1]), q(x)[:1])
    assert (dequantized - linear.weight.float()).abs().max() <= q.scale.float().max() / 2 + 1e-6


def test_quantize_linears_skips_the_head():
    model = nn.Sequential()
    model.add_module("body", nn.Linear(8, 8))
    model.add_module("lm_head", nn.Linear(8, 4))
    assert quantize_linears(model) == 1
    assert isinstance(model.body, Int8WeightOnlyLinear) and isinstance(model.lm_head, nn.Linear)
