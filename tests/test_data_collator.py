"""Unit tests for distillkit.data collation factory and batch collators."""

from types import SimpleNamespace
import torch
import pytest

from distillkit.data import (
    CachedBatchCollator,
    collate_packed_batch,
    create_data_collator,
)
from distillkit.signals import OfflineHiddenStateSignalSource


class DummyTokenizer:
    pad_token_id = 1
    eos_token_id = 2


def test_collate_packed_batch():
    batch = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
        {"input_ids": [4, 5, 6], "labels": [4, 5, 6]},
    ]
    collated = collate_packed_batch(batch)
    assert isinstance(collated["input_ids"], torch.Tensor)
    assert collated["input_ids"].shape == (2, 3)
    assert torch.equal(collated["input_ids"][0], torch.tensor([1, 2, 3]))
    assert torch.equal(collated["labels"][1], torch.tensor([4, 5, 6]))


def test_cached_batch_collator():
    collator = CachedBatchCollator(pad_token_id=0)
    batch = [
        {"doc_id": "doc1", "input_ids": [10, 20, 30], "attention_mask": [1, 1, 1], "labels": [10, 20, 30]},
        {"doc_id": "doc2", "input_ids": [40, 50], "attention_mask": [1, 1], "labels": [40, 50]},
    ]
    collated = collator(batch)
    assert collated["doc_id"] == ["doc1", "doc2"]
    assert collated["input_ids"].shape == (2, 3)
    assert collated["attention_mask"].shape == (2, 3)
    # Second document is padded with 0
    assert collated["input_ids"][1, 2].item() == 0
    assert collated["attention_mask"][1, 2].item() == 0
    assert collated["labels"][1, 2].item() == -100


def test_create_data_collator_cached_signal():
    tokenizer = DummyTokenizer()
    config = SimpleNamespace(dataset=SimpleNamespace(prepacked=False), sidecar=None)
    mock_signal = object.__new__(OfflineHiddenStateSignalSource)
    collator = create_data_collator(config, tokenizer, signal_source=mock_signal)
    assert isinstance(collator, CachedBatchCollator)
    assert collator.pad_token_id == 1


def test_create_data_collator_prepacked():
    tokenizer = DummyTokenizer()
    config = SimpleNamespace(dataset=SimpleNamespace(prepacked=True), sidecar=None)
    collator = create_data_collator(config, tokenizer, signal_source=None)
    assert collator is collate_packed_batch


def test_create_data_collator_default():
    tokenizer = DummyTokenizer()
    config = SimpleNamespace(dataset=SimpleNamespace(prepacked=False), sidecar=None)
    collator = create_data_collator(config, tokenizer, signal_source=None)
    assert collator is None
