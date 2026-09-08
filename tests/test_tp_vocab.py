"""The vocab-parallel tied embedding must be the same function as nn.Embedding and
nn.Linear in both directions, and its composed log-softmax the same number as the
whole row's -- or the memory balance is a silent change to what the run optimizes."""

import pytest
import torch
from torch.nn import functional as F

from distillkit.chunked_head import chunked_head_loss
from distillkit.lossfuncs.kl import sparse_kl_div_inner
from distillkit.tp_vocab import VocabParallelEmbedding, VocabParallelHead

TWO_GPUS = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
DEVICES = pytest.mark.parametrize(
    "devices",
    [["cpu", "cpu"], pytest.param(["cuda:0", "cuda:1"], marks=TWO_GPUS)],
    ids=["cpu", "two_cards"],
)
VOCAB, HIDDEN, TOP_K, PAD = 64, 16, 6, 3


def _pair(devices):
    torch.manual_seed(0)
    source = torch.nn.Embedding(VOCAB, HIDDEN, padding_idx=PAD)
    with torch.no_grad():
        source.weight.normal_()  # the padding row is zero-initialised; make its lookup visible
    embedding = VocabParallelEmbedding(source, devices)
    return source, embedding, VocabParallelHead(embedding)


def _shard_grad(module):
    return torch.cat([shard.grad.cpu() for shard in module.shards])


@DEVICES
def test_lookup_matches_embedding_including_the_padding_gradient(devices):
    source, embedding, head = _pair(devices)
    assert head.shards is embedding.shards, "the tie is the shared ParameterList"
    ids = torch.randint(0, VOCAB, (2, 9))
    ids[0, 0] = PAD
    reference = source(ids)
    reference.square().sum().backward()

    out = embedding(ids.to(devices[0]))
    assert out.device == torch.device(devices[0])
    out.square().sum().backward()

    torch.testing.assert_close(out.cpu(), reference)
    torch.testing.assert_close(_shard_grad(embedding), source.weight.grad)
    assert not _shard_grad(embedding)[PAD].any(), "padding_idx must get no gradient, as in nn.Embedding"


@DEVICES
def test_dense_head_matches_linear(devices):
    source, _, head = _pair(devices)
    hidden = torch.randn(2, 5, HIDDEN)
    out = head(hidden.to(devices[0]))
    assert out.device == torch.device(devices[0])
    torch.testing.assert_close(out.cpu(), F.linear(hidden, source.weight))


@DEVICES
@pytest.mark.parametrize("scale", [None, 0.5])
def test_sharded_logprobs_match_the_whole_row(devices, scale):
    source, _, head = _pair(devices)
    hidden = torch.randn(2, 5, HIDDEN, requires_grad=True)
    ids = torch.randint(0, VOCAB, (2, 5, TOP_K))
    logits = F.linear(hidden, source.weight).float() * (1.0 if scale is None else scale)
    reference = torch.log_softmax(logits, -1).gather(-1, ids)
    reference.sum().backward()

    remote_hidden = hidden.detach().to(devices[0]).requires_grad_(True)
    sharded = head.sharded_logits(remote_hidden)
    assert sharded.shape == (2, 5, VOCAB) and sharded.device == torch.device(devices[0])
    got = sharded.sparse_logprobs(ids.to(devices[0]), scale)
    assert got.device == torch.device(devices[0]) and got.dtype == torch.float32
    got.sum().backward()

    torch.testing.assert_close(got.cpu(), reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(remote_hidden.grad.cpu(), hidden.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(_shard_grad(head), source.weight.grad, rtol=1e-5, atol=1e-6)


def test_truncation_drops_padded_columns():
    source, _, head = _pair(["cpu", "cpu"])
    hidden = torch.randn(1, 4, HIDDEN)
    for vocab_size in (VOCAB, VOCAB - 5, VOCAB // 2, VOCAB // 2 - 7):
        sharded = head.sharded_logits(hidden, vocab_size)
        assert sharded.shape == (1, 4, vocab_size)
        ids = torch.randint(0, vocab_size, (1, 4, TOP_K))
        reference = torch.log_softmax(
            F.linear(hidden, source.weight)[..., :vocab_size], -1
        ).gather(-1, ids)
        torch.testing.assert_close(sharded.sparse_logprobs(ids), reference, rtol=1e-5, atol=1e-6)


@DEVICES
def test_folded_kl_through_the_sharded_head_matches_the_dense_head(devices):
    """The path the trainer uses: chunked_head_loss under checkpoint -> sparse KL."""
    source, _, head = _pair(devices)
    hidden = torch.randn(1, 12, HIDDEN)
    ids = torch.randint(0, VOCAB, (1, 12, TOP_K))
    values = torch.log_softmax(torch.randn(1, 12, TOP_K), -1)
    dense = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
    with torch.no_grad():
        dense.weight.copy_(source.weight)

    reference_hidden = hidden.clone().requires_grad_(True)
    reference = chunked_head_loss(reference_hidden, dense, ids, values, None, 4, sparse_kl_div_inner)
    reference.backward()

    sharded_hidden = hidden.to(devices[0]).requires_grad_(True)
    got = chunked_head_loss(
        sharded_hidden, head, ids.to(devices[0]), values.to(devices[0]), None, 4,
        sparse_kl_div_inner,
    )
    assert got.device == torch.device(devices[0])
    got.backward()

    torch.testing.assert_close(got.cpu(), reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(sharded_hidden.grad.cpu(), reference_hidden.grad, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(_shard_grad(head), dense.weight.grad, rtol=1e-4, atol=1e-6)


def test_uneven_vocabulary_is_refused():
    with pytest.raises(ValueError, match="cannot split 65"):
        VocabParallelEmbedding(torch.nn.Embedding(65, 8), ["cpu", "cpu"])
