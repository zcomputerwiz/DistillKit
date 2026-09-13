"""Backward compatibility shim for distillkit.core.sortish_sampler."""

from distillkit.core.sortish_sampler import (
    DEFAULT_SORT_WINDOW,
    SortishSampler,
    sortish_indices,
)

__all__ = ["DEFAULT_SORT_WINDOW", "SortishSampler", "sortish_indices"]
