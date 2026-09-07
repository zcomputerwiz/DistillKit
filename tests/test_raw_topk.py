"""Uncompressed offline signals must work without configuring a codec."""
import pytest
import torch

from distillkit.compression import LogprobCompressor
from distillkit.signals import OfflineSignalSource


def test_uncompressed_offline_signal_preserves_large_token_ids():
    ids = torch.tensor([[[248319, 65536], [248044, 17]]])
    values = torch.tensor([[[-0.2, -2.0], [-0.3, -1.5]]])
    source = OfflineSignalSource(LogprobCompressor(), vocab_size=248320)
    signal = source.get_signal({"token_ids": ids, "top_values": values})
    assert torch.equal(signal.sparse_ids, ids)
    assert torch.equal(signal.sparse_values, values)
    assert signal.log_values and signal.hidden_states is None
    assert signal.vocab_size == 248320


def test_compression_requires_configuration():
    codec = LogprobCompressor()
    with pytest.raises(ValueError, match="No config"):
        codec.compress_from_sparse(torch.tensor([[1]]), torch.tensor([[-0.5]]))
    with pytest.raises(ValueError, match="No config"):
        codec.compress(torch.tensor([[-1.0, -0.5]]))
