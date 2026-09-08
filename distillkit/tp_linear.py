"""Column- and row-parallel linear layers over the single-process collectives.

The two halves of a tensor-parallel block, in transformers' own vocabulary
(``base_model_tp_plan`` marks q/k/v/gate/up ``colwise`` and o_proj/down_proj
``rowwise``):

* **Column-parallel** splits the *output* channels. Every shard needs the whole
  input, so the input is replicated and the outputs stay split -- no reduction.
* **Row-parallel** splits the *input* channels, consuming a split input and
  producing partial sums of the whole output, which one all-reduce completes.

Chaining them is why an MLP costs a single reduction: ``(x @ W1) @ W2`` with W1 split
by columns and W2 by rows is exactly ``sum_i (x @ W1_i) @ W2_i``, and the
intermediate never has to be gathered.

Shards are ordinary ``nn.Parameter`` on their own device, so gradients, the optimizer
and its state all follow the shard without further arrangement -- which is how tensor
parallelism subsumes ZeRO-2's memory saving here rather than needing it as well.
"""

from __future__ import annotations

import torch
from torch import nn

from distillkit.tensor_parallel import all_reduce, reduce_to, replicate


def _devices(devices) -> list[torch.device]:
    resolved = [torch.device(d) for d in devices]
    if len(resolved) < 1:
        raise ValueError("Tensor parallelism needs at least one device")
    return resolved


def split_sizes(total: int, parts: int) -> list[int]:
    """Even split, rejecting anything that would silently reshape the maths.

    Head counts and channel widths in this architecture all divide by two; an
    uneven split would still run and would change what each rank computes.
    """
    if total % parts:
        raise ValueError(
            f"cannot split {total} evenly across {parts} devices; shard by a "
            f"dimension that divides, or replicate this module instead"
        )
    return [total // parts] * parts


class ColumnParallelLinear(nn.Module):
    """``y_i = x @ W_i^T`` with the output channels split across devices.

    Returns one tensor per device. ``gather_output=True`` concatenates them back --
    which upstream's plan uses for the linear-attention projections, and which costs
    a transfer per device, so prefer feeding a row-parallel layer instead.
    """

    def __init__(self, source: nn.Linear, devices, gather_output: bool = False):
        super().__init__()
        self.devices = _devices(devices)
        self.gather_output = gather_output
        self.in_features = source.in_features
        self.out_features = source.out_features
        sizes = split_sizes(source.out_features, len(self.devices))
        weights, biases = [], []
        offset = 0
        for size, device in zip(sizes, self.devices):
            piece = slice(offset, offset + size)
            weights.append(nn.Parameter(source.weight.data[piece].detach().clone().to(device)))
            if source.bias is not None:
                biases.append(nn.Parameter(source.bias.data[piece].detach().clone().to(device)))
            offset += size
        self.shards = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases) if biases else None

    def forward(self, x, copies=None):
        """``copies`` lets a caller replicate once and feed several projections.

        Two column-parallel layers on the same input would otherwise each create a
        Replicate node, and each returns the *input tensor itself* for the home
        device -- two autograd outputs aliasing one tensor. Combining those aliases
        elementwise, as an MLP does with gate and up, made non-reentrant
        checkpointing recompute different values. Sharing one replication avoids
        that and costs one fewer node.
        """
        if copies is None:
            copies = replicate(x, self.devices)
        outputs = [
            nn.functional.linear(
                copy, weight, None if self.biases is None else self.biases[index]
            )
            for index, (copy, weight) in enumerate(zip(copies, self.shards))
        ]
        if not self.gather_output:
            return outputs
        home = outputs[0].device
        return torch.cat([out.to(home) for out in outputs], dim=-1)


class RowParallelLinear(nn.Module):
    """``y = sum_i x_i @ W_i^T`` with the input channels split across devices.

    Takes the per-device list a column-parallel layer produced and returns one
    reduced tensor per device. The bias is added once, on the first device, because
    every device already holds the full sum.
    """

    def __init__(self, source: nn.Linear, devices, reduce_only: bool = False):
        super().__init__()
        self.devices = _devices(devices)
        # reduce_only: the consumer is the home device alone, so skip copying the
        # total back out to the others. That is the case throughout this model,
        # where the residual stream stays on the home card.
        self.reduce_only = reduce_only
        self.in_features = source.in_features
        self.out_features = source.out_features
        sizes = split_sizes(source.in_features, len(self.devices))
        weights = []
        offset = 0
        for size, device in zip(sizes, self.devices):
            piece = slice(offset, offset + size)
            weights.append(
                nn.Parameter(source.weight.data[:, piece].detach().clone().to(device))
            )
            offset += size
        self.shards = nn.ParameterList(weights)
        self.bias = (
            nn.Parameter(source.bias.data.detach().clone().to(self.devices[0]))
            if source.bias is not None
            else None
        )

    def forward(self, parts):
        if isinstance(parts, torch.Tensor):
            parts = list(torch.split(parts, [w.shape[1] for w in self.shards], dim=-1))
            parts = [p.to(d) for p, d in zip(parts, self.devices)]
        partials = [
            nn.functional.linear(part, weight)
            for part, weight in zip(parts, self.shards)
        ]
        if self.reduce_only:
            total = reduce_to(partials, self.devices[0])
            return [total + self.bias if self.bias is not None else total]
        reduced = list(all_reduce(partials))
        if self.bias is not None:
            # Added after the reduction, once: each device holds the same total, so
            # adding it per shard would multiply it by the device count.
            reduced = [
                out + self.bias.to(out.device) if index else out + self.bias
                for index, out in enumerate(reduced)
            ]
        return reduced
