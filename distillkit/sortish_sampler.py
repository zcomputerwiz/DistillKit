"""Length grouping that keeps the padding saving without the ordering cost.

`train_sampling_strategy: group_by_length` made a stage-2 epoch 1.58x faster and
`eval_loss` 0.0176 worse. A control isolated the cause: grouping at batch 1, where no
padding exists and grouping therefore changes *nothing but the order*, cost 0.0145 of
that 0.0176. So the loss is an ordering effect, not a batching or padding one, and it is
not noise -- with the mathematics held fixed this pipeline reproduces `eval_loss` to four
decimals (the layer split and tensor parallelism gave 0.5330 and 0.5329).

Two properties of HF's sampler cause it, and neither is needed to save padding.

**It groups at the optimizer step, not the microbatch.** ``Trainer._get_train_sampler``
constructs ``LengthGroupedSampler(train_batch_size * gradient_accumulation_steps)``, so
with accumulation 16 every *step* sees 16 near-identical lengths. Padding is a property
of a single forward pass: only the microbatch has to be homogeneous. Grouping the whole
step additionally strips the length diversity out of each optimizer update, which is the
failure mode reported in the data-organization literature -- samples sharing nearly
identical attributes give correlated gradients and bias the update.

**It emits batches in descending length within each megabatch.** The lengths sawtooth
over the epoch (transformers#20810, still open), so the model sees a repeating long-to-
short curriculum rather than a stationary sample. fastai's ``SortishSampler`` and the
bucketing literature both shuffle the batch order after sorting; HF sorts and does not.

This sampler keeps everything that works and changes only those two things: it groups at
``batch_size`` -- pass the microbatch size -- and shuffles the batch order, retaining
HF's placement of the longest batch first so an over-budget shape still fails on step 0
rather than an hour in. Padding stays as low as the microbatch allows, because that is
the only place padding is decided.
"""

from __future__ import annotations

import torch
from torch.utils.data import Sampler
from transformers.trainer_pt_utils import get_length_grouped_indices

__all__ = ["SortishSampler", "sortish_indices"]


def sortish_indices(lengths, batch_size, mega_batch_mult=None, generator=None) -> list[int]:
    """HF's length grouping with the batch order shuffled, longest batch still first."""
    indices = get_length_grouped_indices(
        lengths, batch_size, mega_batch_mult=mega_batch_mult, generator=generator
    )
    batches = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]
    if len(batches) <= 2:
        return [index for batch in batches for index in batch]
    # batches[0] holds the longest element -- HF swaps it there so an OOM happens on the
    # first step. Keep that and shuffle the rest.
    order = torch.randperm(len(batches) - 1, generator=generator).tolist()
    shuffled = [batches[0]] + [batches[position + 1] for position in order]
    return [index for batch in shuffled for index in batch]


class SortishSampler(Sampler):
    """``LengthGroupedSampler`` with the batch order shuffled.

    ``batch_size`` should be the **microbatch** size (``args.train_batch_size``), not the
    effective batch: padding is decided per forward pass, and grouping any wider only
    removes length diversity from each optimizer step. See the module docstring.
    """

    def __init__(self, batch_size: int, lengths, generator=None):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if lengths is None:
            raise ValueError(
                "sortish batching needs a length per example; the offline cache exposes "
                "a `length` column for exactly this"
            )
        self.batch_size = batch_size
        self.lengths = list(lengths)
        self.generator = generator

    def __len__(self) -> int:
        return len(self.lengths)

    def __iter__(self):
        return iter(sortish_indices(self.lengths, self.batch_size, generator=self.generator))
