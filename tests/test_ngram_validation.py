"""Malformed artifacts must fail before producing plausible but unrelated rows."""
import numpy as np
import pytest
import torch

from distillkit.ngram_table import (
    GGUFNGramTable, IQ4NLDequant, NGramTableSpec, dequantize_iq4nl_rows,
)


def write_table(path, quant_name="IQ4_NL"):
    gguf = pytest.importorskip("gguf")
    raw = np.zeros((4, 90), dtype=np.uint8)
    blocks = raw.reshape(4, 5, 18)
    blocks[..., :2] = np.array([0.125], dtype=np.float16).view(np.uint8)
    blocks[..., 2:] = np.arange(16, dtype=np.uint8)
    writer = gguf.GGUFWriter(str(path), "qwen4exp")
    writer.add_tensor("per_layer_token_embd.weight", raw,
                      raw_dtype=getattr(gguf.GGMLQuantizationType, quant_name))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return raw


def test_rejects_same_size_wrong_codebook(tmp_path):
    path = tmp_path / "wrong.gguf"
    write_table(path, "Q4_0")
    with pytest.raises(ValueError, match="must be IQ4_NL"):
        GGUFNGramTable(str(path), spec=NGramTableSpec(4, 4))


def test_small_real_gguf_gather_and_residency(tmp_path):
    path = tmp_path / "table.gguf"
    raw = write_table(path)
    table = GGUFNGramTable(str(path), spec=NGramTableSpec(4, 4))
    rows = np.array([[3, 0], [1, 3]])
    assert np.array_equal(table.gather_raw(rows), raw[rows])
    expected = dequantize_iq4nl_rows(raw[rows])
    assert np.array_equal(table.gather(rows), expected)
    assert table.gather_raw(np.array([], dtype=np.int64)).shape == (0, 90)
    assert table.gather(np.array([], dtype=np.int64)).shape == (0, 160)
    for bad in ([-1], [4], np.array([2**64 - 1], dtype=np.uint64)):
        with pytest.raises(IndexError):
            table.gather_raw(bad)
    for bad in ([1.5], [True], ["1"]):
        with pytest.raises(TypeError):
            table.gather_raw(bad)
    with pytest.raises(ValueError):
        table.prefault(0)
    table.load_resident()
    assert table.resident
    assert np.array_equal(table.gather(rows), expected)
    assert table.load_resident() == 0


@pytest.mark.parametrize("kwargs", [
    {"head_dim": 31}, {"head_dim": 0}, {"n_rows": 0}, {"n_real_rows": 0},
    {"n_rows": 1, "n_real_rows": 2},
])
def test_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        NGramTableSpec(**kwargs)


@pytest.mark.parametrize("shape", [(), (2, 0), (2, 19)])
def test_rejects_incomplete_blocks(shape):
    raw = np.zeros(shape, dtype=np.uint8)
    with pytest.raises(ValueError):
        dequantize_iq4nl_rows(raw)
    with pytest.raises(ValueError):
        IQ4NLDequant()(torch.from_numpy(raw))


def test_numpy_rejects_non_bytes():
    with pytest.raises(TypeError):
        dequantize_iq4nl_rows(np.zeros((1, 90), dtype=np.float32))
