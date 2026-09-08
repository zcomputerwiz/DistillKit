"""Sortish batching must keep the padding saving and drop the ordering.

The measured cost of `group_by_length` on a stage-2 epoch was 0.0176 of eval_loss, of
which a batch-1 control attributed 0.0145 to ordering alone. So these check both halves:
padding stays near the grouped sampler's, and the descending-length curriculum is gone.
"""

import random

import pytest
import torch

from distillkit.sortish_sampler import SortishSampler, sortish_indices

BATCH = 4


def _lengths(count=1200, seed=0):
    # Shaped like the real corpus: median ~550, a long tail to 4096.
    rng = random.Random(seed)
    return [min(4096, max(145, int(rng.lognormvariate(6.3, 0.75)))) for _ in range(count)]


def _generator(seed=0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _batches(indices, lengths, batch_size=BATCH):
    return [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]


def _padding_waste(indices, lengths, batch_size=BATCH):
    real = padded = 0
    for batch in _batches(indices, lengths, batch_size):
        padded += len(batch) * max(lengths[i] for i in batch)
        real += sum(lengths[i] for i in batch)
    return (padded - real) / real


def test_every_example_appears_exactly_once():
    lengths = _lengths()
    indices = sortish_indices(lengths, BATCH, generator=_generator())
    assert sorted(indices) == list(range(len(lengths)))


def test_padding_stays_as_low_as_the_grouped_sampler():
    """Shuffling batch *order* must not disturb what is inside a batch."""
    from transformers.trainer_pt_utils import get_length_grouped_indices

    lengths = _lengths()
    grouped = get_length_grouped_indices(lengths, BATCH, generator=_generator())
    sortish = sortish_indices(lengths, BATCH, generator=_generator())

    random_order = list(range(len(lengths)))
    random.Random(1).shuffle(random_order)

    assert _padding_waste(sortish, lengths) < 0.05
    assert _padding_waste(sortish, lengths) <= _padding_waste(grouped, lengths) * 1.05
    assert _padding_waste(random_order, lengths) > 5 * _padding_waste(sortish, lengths)


def test_batch_order_is_not_a_descending_curriculum():
    """The whole point: grouped sampling sawtooths, this must not."""
    from transformers.trainer_pt_utils import get_length_grouped_indices

    lengths = _lengths()

    def descending_fraction(indices):
        # Skip the deliberately-first longest batch in both, and compare each batch's
        # max length with the next one's.
        maxima = [max(lengths[i] for i in batch)
                  for batch in _batches(indices, lengths)][1:]
        drops = sum(a > b for a, b in zip(maxima, maxima[1:]))
        return drops / (len(maxima) - 1)

    grouped = descending_fraction(get_length_grouped_indices(lengths, BATCH, generator=_generator()))
    sortish = descending_fraction(sortish_indices(lengths, BATCH, generator=_generator()))
    assert grouped > 0.9, f"expected the grouped sampler to descend, got {grouped:.2f}"
    assert 0.35 < sortish < 0.65, f"sortish should look like a coin flip, got {sortish:.2f}"


def test_longest_batch_still_comes_first():
    """HF puts it there so an over-budget shape OOMs on step 0, not an hour in."""
    lengths = _lengths()
    indices = sortish_indices(lengths, BATCH, generator=_generator())
    batches = _batches(indices, lengths)
    assert max(lengths[i] for i in batches[0]) == max(lengths)


def test_same_seed_gives_the_same_order():
    lengths = _lengths()
    first = sortish_indices(lengths, BATCH, generator=_generator(7))
    second = sortish_indices(lengths, BATCH, generator=_generator(7))
    third = sortish_indices(lengths, BATCH, generator=_generator(8))
    assert first == second
    assert first != third


@pytest.mark.parametrize("count", [1, 4, 5, 8, 9])
def test_short_datasets_are_still_a_permutation(count):
    lengths = _lengths(count=count)
    indices = sortish_indices(lengths, BATCH, generator=_generator())
    assert sorted(indices) == list(range(count))


@pytest.mark.parametrize("count", [1150, 1151, 1153, 999])
def test_a_short_final_batch_stays_final(count):
    """The flat stream is re-chunked at batch_size by the DataLoader, so a short batch
    anywhere but the end shifts every later boundary and undoes the grouping.

    Caught by adversarial review, not by these tests: the original fixtures all divided
    evenly by the batch size, which is exactly the case that hides it.
    """
    lengths = _lengths(count=count)
    indices = sortish_indices(lengths, BATCH, generator=_generator())
    assert sorted(indices) == list(range(count))
    assert _padding_waste(indices, lengths) < 0.05, "a misplaced short batch mixes lengths"


def test_the_sort_window_is_independent_of_the_batch_size():
    """Padding is set by how many examples are sorted together, which HF ties to the
    batch size. Grouping at the microbatch must not cost padding to do it."""
    from transformers.trainer_pt_utils import get_length_grouped_indices

    lengths = _lengths()
    # What HF would do grouping at the optimizer step (batch 4 x accumulation 4).
    hf = _padding_waste(get_length_grouped_indices(lengths, 16, generator=_generator()), lengths)
    narrow = _padding_waste(sortish_indices(lengths, BATCH, sort_window=200, generator=_generator()), lengths)
    wide = _padding_waste(sortish_indices(lengths, BATCH, generator=_generator()), lengths)

    assert narrow > hf, "the narrow window should be the regression this guards against"
    assert wide < hf, f"sortish {wide:.4f} should beat HF's grouping {hf:.4f}"


def test_sampler_wraps_the_indices():
    lengths = _lengths(200)
    sampler = SortishSampler(BATCH, lengths, generator=_generator())
    assert len(sampler) == 200
    assert sorted(sampler) == list(range(200))


def test_sampler_rejects_missing_lengths():
    with pytest.raises(ValueError, match="needs a length per example"):
        SortishSampler(BATCH, None)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        SortishSampler(0, [1, 2, 3])
