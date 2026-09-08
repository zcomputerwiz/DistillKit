"""Sharded linears must be arithmetically indistinguishable from the originals.

Tensor parallelism is only safe if `shard(f)(x) == f(x)` to floating-point noise, in
both directions. These compare against the unsharded `nn.Linear` autograd gives, since
a sharding error produces a plausible number rather than an exception.
"""

import pytest
import torch
from torch import nn

from distillkit.tp_linear import ColumnParallelLinear, RowParallelLinear, split_sizes

TWO_GPUS = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs two CUDA devices"
)


def test_split_sizes_rejects_an_uneven_split():
    """An uneven split still runs and silently changes what each rank computes."""
    assert split_sizes(8, 2) == [4, 4]
    with pytest.raises(ValueError, match="evenly"):
        split_sizes(9, 2)


@pytest.mark.parametrize("bias", [False, True])
def test_column_parallel_matches_the_original_on_cpu(bias):
    torch.manual_seed(0)
    source = nn.Linear(16, 8, bias=bias)
    x = torch.randn(4, 16, requires_grad=True)
    reference = source(x)

    layer = ColumnParallelLinear(source, ["cpu", "cpu"])
    parts = layer(x)
    assert [p.shape[-1] for p in parts] == [4, 4]
    torch.testing.assert_close(torch.cat(parts, dim=-1), reference)


@pytest.mark.parametrize("bias", [False, True])
def test_column_then_row_matches_the_original_on_cpu(bias):
    """The MLP pattern: no gather between the two halves."""
    torch.manual_seed(0)
    up = nn.Linear(16, 32, bias=bias)
    down = nn.Linear(32, 16, bias=bias)
    x = torch.randn(4, 16, requires_grad=True)
    reference = down(up(x))
    reference.sum().backward()
    reference_grad = x.grad.clone()

    sharded_x = x.detach().clone().requires_grad_(True)
    column = ColumnParallelLinear(up, ["cpu", "cpu"])
    row = RowParallelLinear(down, ["cpu", "cpu"])
    out = row(column(sharded_x))[0]
    out.sum().backward()

    torch.testing.assert_close(out, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(sharded_x.grad, reference_grad, rtol=1e-5, atol=1e-6)


def test_row_parallel_bias_is_added_once_not_per_shard():
    """Every device holds the full sum after the reduction.

    Adding the bias inside each shard would multiply it by the device count -- a
    bug that survives every shape check and just shifts the output.
    """
    torch.manual_seed(0)
    down = nn.Linear(8, 4, bias=True)
    with torch.no_grad():
        down.weight.zero_()
        down.bias.fill_(3.0)
    row = RowParallelLinear(down, ["cpu", "cpu"])
    parts = [torch.randn(2, 4), torch.randn(2, 4)]
    out = row(parts)
    for shard_out in out:
        torch.testing.assert_close(shard_out, torch.full((2, 4), 3.0))


def test_weight_gradients_match_their_slice_of_the_original():
    torch.manual_seed(0)
    source = nn.Linear(16, 8, bias=False)
    x = torch.randn(4, 16)

    reference = nn.Linear(16, 8, bias=False)
    reference.load_state_dict(source.state_dict())
    reference(x).sum().backward()

    layer = ColumnParallelLinear(source, ["cpu", "cpu"])
    torch.cat(layer(x), dim=-1).sum().backward()
    for index, shard in enumerate(layer.shards):
        piece = slice(index * 4, (index + 1) * 4)
        torch.testing.assert_close(shard.grad, reference.weight.grad[piece])


@TWO_GPUS
def test_shards_live_on_their_own_devices_with_their_optimizer_state():
    """Sharded parameters carry gradients and optimizer state to their own card.

    This is why tensor parallelism subsumes ZeRO-2's saving here rather than needing
    it too, so it is worth pinning rather than assuming.
    """
    source = nn.Linear(64, 32, bias=True)
    layer = ColumnParallelLinear(source, ["cuda:0", "cuda:1"])
    assert layer.shards[0].device == torch.device("cuda", 0)
    assert layer.shards[1].device == torch.device("cuda", 1)

    x = torch.randn(8, 64, device="cuda:0")
    torch.cat([p.to("cuda:0") for p in layer(x)], dim=-1).sum().backward()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    optimizer.step()
    for parameter in layer.parameters():
        assert parameter.grad.device == parameter.device
        for value in optimizer.state[parameter].values():
            if torch.is_tensor(value) and value.dim():
                assert value.device == parameter.device


@TWO_GPUS
def test_column_then_row_matches_the_original_across_two_cards():
    torch.manual_seed(0)
    up = nn.Linear(64, 128, bias=False)
    down = nn.Linear(128, 64, bias=False)
    x = torch.randn(16, 64)

    reference_x = x.clone().to("cuda:0").requires_grad_(True)
    reference = down.to("cuda:0")(up.to("cuda:0")(reference_x))
    reference.sum().backward()

    up_cpu, down_cpu = up.to("cpu"), down.to("cpu")
    sharded_x = x.clone().to("cuda:0").requires_grad_(True)
    column = ColumnParallelLinear(up_cpu, ["cuda:0", "cuda:1"])
    row = RowParallelLinear(down_cpu, ["cuda:0", "cuda:1"])
    out = row(column(sharded_x))[0]
    out.sum().backward()

    torch.testing.assert_close(out, reference, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(sharded_x.grad, reference_x.grad, rtol=1e-4, atol=1e-5)
