import pickle

import numpy as np
import pytest
import torch

from distillkit.ngram_hash import NGramHashConfig, NGramHasher
from distillkit.ngram_table import GGUFNGramTable, NGramTableSpec
from distillkit.sidecar_collator import SidecarDataCollator


class TinyTable:
    def __init__(self, hasher):
        self.spec = NGramTableSpec(hasher.padded_vocab_size, hasher.total_vocab_size, 32)
        self.raw = np.arange(self.spec.n_bytes, dtype=np.uint8).reshape(self.spec.n_rows, -1)

    def gather_raw(self, rows):
        return self.raw[rows.numpy()]


def make_hasher():
    return NGramHasher(NGramHashConfig(vocab_size=64, ngram_size=2, heads_per_ngram=2,
        ngram_vocab_size_base=17, make_ngram_vocab_size_divisible_by=8, ple_embed_dim=64, eos_token_id=3))


def identity_collate(features):
    return features


@pytest.mark.parametrize("left_padding", [True, False])
def test_collator_preserves_signals_and_padding_boundaries(left_padding):
    hasher = make_hasher()
    table = TinyTable(hasher)
    collator = SidecarDataCollator(identity_collate, table, hasher)
    tokens = [5, 7, 3, 12]
    padded = [0, 0] + tokens if left_padding else tokens + [0, 0]
    mask = [0, 0, 1, 1, 1, 1] if left_padding else [1, 1, 1, 1, 0, 0]
    teacher = object()
    inputs = {"input_ids": torch.tensor([padded]), "attention_mask": torch.tensor([mask]),
        "labels": torch.tensor([padded]), "teacher_signal": teacher, "doc_id": ["document-7"]}
    result = collator(inputs)
    assert "ngram_raw" not in inputs
    assert result["teacher_signal"] is teacher
    assert result["doc_id"] == ["document-7"]
    assert result["input_ids"] is inputs["input_ids"]
    expected = table.gather_raw(hasher.row_indices(torch.tensor([tokens])))
    actual = result["ngram_raw"][:, 2:] if left_padding else result["ngram_raw"][:, :-2]
    assert np.array_equal(actual.numpy(), expected)
    assert result["ngram_raw"].dtype == torch.uint8
    assert result["ngram_raw"].shape == (1, 6, 2, 18)


def test_mapped_table_pickling_keeps_only_reopen_metadata():
    hasher = make_hasher()
    table = GGUFNGramTable.__new__(GGUFNGramTable)
    table.spec = NGramTableSpec(hasher.padded_vocab_size, hasher.total_vocab_size, 32)
    table.gguf_path = "placeholder.gguf"
    table.tensor_name = "per_layer_token_embd.weight"
    table.resident = False
    table.raw = np.zeros((1_000_000,), dtype=np.uint8)
    collator = SidecarDataCollator(identity_collate, table, hasher)
    serialized = pickle.dumps(collator)
    assert len(serialized) < 10000
    reloaded = pickle.loads(serialized)
    assert reloaded.table is None
    assert reloaded._table_factory["gguf_path"] == table.gguf_path
    table.resident = True
    with pytest.raises(RuntimeError, match="num_workers=0"):
        pickle.dumps(collator)


def test_bad_tokens_and_packing_rejected():
    hasher = make_hasher()
    collator = SidecarDataCollator(identity_collate, TinyTable(hasher), hasher)
    with pytest.raises(ValueError, match="vocabulary"):
        collator({"input_ids": torch.tensor([[64]])})
    with pytest.raises(ValueError, match="binary"):
        collator({"input_ids": torch.tensor([[4, 5]]), "attention_mask": torch.tensor([[1, 2]])})
    with pytest.raises(ValueError, match="integer"):
        collator({"input_ids": torch.tensor([[4.0]])})
