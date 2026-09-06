"""Bit-identity gate for the vendored n-gram row indexer.

``distillkit/ngram_hash.py`` is a hand-copied port of the index half of
``Qwen4ExpTextNGramEmbedding``. These tests run the *live* transformers reference
and assert the two agree exactly. If transformers changes the hash, these fail --
which is the point. A silently-diverged hash fetches rows unrelated to the
intended n-gram and is otherwise invisible.

The reference's ``forward`` ends by indexing a 320,001,536 x 160 embedding
(51.2 GB), so we swap ``nn.Embedding`` for a stub that returns the ids themselves.
That keeps the reference's *real* forward path under test -- shift, mix, XOR fold,
remainder, offset, concat, tail-slice -- rather than a re-derivation of it.
"""

import json
import os

import pytest
import torch
import torch.nn as nn

from distillkit.ngram_hash import (
    FLASH_NEXT_NGRAM_CONFIG,
    NGramHashConfig,
    NGramHasher,
    build_layer_multipliers,
    find_nth_prime_after,
    splitmix64,
)

qwen4_exp = pytest.importorskip(
    "transformers.models.qwen4_exp.modeling_qwen4_exp",
    reason="installed transformers has no qwen4_exp reference implementation",
)

FLASH_NEXT_CONFIG_JSON = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "huggingface",
    "hub",
    "models--Qwen--Qwen3.8-Flash-Next",
    "snapshots",
    "de4b8e4d43b917e7706784d8bb445c9af86a3540",
    "config.json",
)


class _IdentityEmbedding(nn.Module):
    """Stands in for the 51.2 GB table; returns the row ids so we can read them out.

    The reference does ``self.ngram_embedding(ids).to(...).flatten(-2)``, so returning
    ``ids.unsqueeze(-1)`` makes ``forward`` yield exactly ``[batch, seq, ngram_heads]``
    row indices.
    """

    def __init__(self, num_embeddings, embedding_dim, *args, **kwargs):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return ids.unsqueeze(-1)


@pytest.fixture(scope="module")
def flash_next_text_config():
    """The real Flash-Next text config -- the authoritative source for every constant."""
    if not os.path.exists(FLASH_NEXT_CONFIG_JSON):
        pytest.skip(f"Flash-Next config.json not cached at {FLASH_NEXT_CONFIG_JSON}")
    with open(FLASH_NEXT_CONFIG_JSON) as f:
        raw = json.load(f)
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    return Qwen4ExpTextConfig(**raw["text_config"])


@pytest.fixture(scope="module")
def reference_module(flash_next_text_config, module_mocker=None):
    """The live reference n-gram embedding, with the giant table stubbed out."""
    real_embedding = nn.Embedding
    nn.Embedding = _IdentityEmbedding
    try:
        module = qwen4_exp.Qwen4ExpTextNGramEmbedding(
            flash_next_text_config,
            embedding_dim=flash_next_text_config.ple_embed_dim,
            layer_idx=1,
            ple_layer_index=0,
        )
    finally:
        nn.Embedding = real_embedding
    return module.eval()


@pytest.fixture(scope="module")
def ported_hasher(flash_next_text_config):
    return NGramHasher(
        NGramHashConfig.from_pretrained_config(flash_next_text_config, ple_layer_index=0)
    )


# --- constants ---------------------------------------------------------------


def test_splitmix64_matches_reference():
    for value in [0, 1, 1234, 2**31, 2**63 - 1, (1 << 64) - 1]:
        assert splitmix64(value) == qwen4_exp._splitmix64(value)


def test_prime_search_matches_reference():
    for count in range(1, 17):
        assert find_nth_prime_after(19_999_999, count) == qwen4_exp._find_nth_prime_after(
            19_999_999, count
        )


def test_layer_multipliers_match_reference():
    ours = build_layer_multipliers(248320, 3, 0, 1234)
    theirs = qwen4_exp._build_layer_multipliers(248320, 3, 0, 1234)
    assert torch.equal(ours, theirs)


def test_derived_table_geometry(ported_hasher):
    """The numbers the whole sizing plan rests on."""
    assert ported_hasher.config.ngram_heads == 16
    assert ported_hasher.config.head_dim == 160
    assert ported_hasher.total_vocab_size == 320_001_446
    assert ported_hasher.padded_vocab_size == 320_001_536
    assert ported_hasher.head_offsets[0].item() == 0
    assert ported_hasher.head_offsets[-1].item() == 300_001_275
    # 51.2 GB at fp8, 102.4 GB at bf16 -- the figure the memory plan is built on.
    assert ported_hasher.padded_vocab_size * 160 == 51_200_245_760


def test_buffers_match_reference(ported_hasher, reference_module):
    assert torch.equal(ported_hasher.head_vocab_sizes, reference_module.ngram_heads_vocab_sizes)
    assert torch.equal(ported_hasher.head_offsets, reference_module.ngram_heads_offsets)
    assert torch.equal(ported_hasher.layer_multipliers, reference_module.layer_multipliers)
    assert ported_hasher.total_vocab_size == reference_module.total_vocab_size


# --- the actual index parity gate --------------------------------------------


@pytest.mark.parametrize("batch,seq", [(1, 1), (1, 16), (4, 128), (2, 4096), (3, 7)])
def test_row_indices_bit_identical(ported_hasher, reference_module, batch, seq):
    generator = torch.Generator().manual_seed(batch * 1000 + seq)
    input_ids = torch.randint(
        0, FLASH_NEXT_NGRAM_CONFIG.vocab_size, (batch, seq), generator=generator, dtype=torch.long
    )
    expected = reference_module(input_ids, past_key_values=None)
    actual = ported_hasher.row_indices(input_ids)
    assert actual.shape == (batch, seq, 16)
    assert torch.equal(actual, expected), "vendored n-gram hash diverged from transformers"


def test_row_indices_with_eos_boundaries(ported_hasher, reference_module):
    """EOS resets the n-gram context; the segment logic is the subtlest part of the port."""
    eos = FLASH_NEXT_NGRAM_CONFIG.eos_token_id
    input_ids = torch.tensor(
        [
            [10, 20, eos, 30, 40, 50, eos, eos, 60, 70],
            [eos, 1, 2, 3, eos, 4, 5, 6, 7, eos],
        ],
        dtype=torch.long,
    )
    expected = reference_module(input_ids, past_key_values=None)
    actual = ported_hasher.row_indices(input_ids)
    assert torch.equal(actual, expected)


def test_row_indices_extreme_token_ids(ported_hasher, reference_module):
    """Max-vocab ids drive the int64 multiply closest to the ceiling."""
    vocab = FLASH_NEXT_NGRAM_CONFIG.vocab_size
    input_ids = torch.tensor(
        [[0, vocab - 1, vocab - 1, 0, vocab - 1, 1, vocab - 2, vocab - 1]], dtype=torch.long
    )
    expected = reference_module(input_ids, past_key_values=None)
    actual = ported_hasher.row_indices(input_ids)
    assert torch.equal(actual, expected)


def test_multiply_does_not_overflow_int64():
    """The multipliers are constructed so token_id * multiplier cannot wrap. Prove it."""
    cfg = FLASH_NEXT_NGRAM_CONFIG
    multipliers = build_layer_multipliers(cfg.vocab_size, cfg.ngram_size, 0, cfg.seed)
    max_product = (cfg.vocab_size - 1) * int(multipliers.max())
    assert max_product < 2**63, "int64 multiply would wrap -- rows would be garbage"


def test_xor_fold_never_sets_the_sign_bit(ported_hasher):
    """The multiplier bound -- not the choice of remainder -- is what keeps rows valid.

    ``build_layer_multipliers`` caps every multiplier at ``(2**63-1)//vocab_size``, so
    each ``token_id * multiplier`` has its sign bit clear, and XOR-ing such values keeps
    it clear. Rows are therefore non-negative before the modulo ever runs.

    This is worth pinning explicitly: it is tempting to read the near-ceiling multiply as
    "relies on signed overflow". It does not, and must not -- a wrapped product would
    change the hash and every fetched row with it.
    """
    cfg = ported_hasher.config
    generator = torch.Generator().manual_seed(7)
    input_ids = torch.randint(0, cfg.vocab_size, (8, 512), generator=generator, dtype=torch.long)

    multipliers = ported_hasher.layer_multipliers
    shifted = [ported_hasher._shift_right_ignore_eos(input_ids, s) for s in range(cfg.ngram_size)]
    products = [shifted[p] * multipliers[p] for p in range(cfg.ngram_size)]
    for position, product in enumerate(products):
        assert product.min() >= 0, f"multiply for shift {position} wrapped into the sign bit"

    mixed = products[0]
    for position in range(1, 3):
        mixed = torch.bitwise_xor(mixed, products[position])
    assert mixed.min() >= 0, "XOR fold set the sign bit -- multiplier bound violated"

    rows = ported_hasher.row_indices(input_ids)
    assert rows.min() >= 0
    assert rows.max() < ported_hasher.padded_vocab_size


def test_previous_context_matches_reference_default(ported_hasher, reference_module):
    """Passing an explicit EOS context must equal the reference's implicit default."""
    generator = torch.Generator().manual_seed(99)
    input_ids = torch.randint(0, 248320, (2, 64), generator=generator, dtype=torch.long)
    explicit = torch.full(
        (2, FLASH_NEXT_NGRAM_CONFIG.context_len),
        FLASH_NEXT_NGRAM_CONFIG.eos_token_id,
        dtype=torch.long,
    )
    assert torch.equal(
        ported_hasher.row_indices(input_ids, previous_context=explicit),
        reference_module(input_ids, past_key_values=None),
    )


def test_head_blocks_are_disjoint_and_ordered(ported_hasher):
    """Heads 0-7 are bigram, 8-15 trigram; each head's rows live in its own band.

    The 2560-dim sidecar vector is the 16 head embeddings concatenated in this order,
    so a permuted head order would scramble the projection input.
    """
    generator = torch.Generator().manual_seed(3)
    input_ids = torch.randint(0, 248320, (4, 256), generator=generator, dtype=torch.long)
    rows = ported_hasher.row_indices(input_ids)
    for head in range(16):
        low = ported_hasher.head_offsets[head].item()
        high = low + ported_hasher.head_vocab_sizes[head].item()
        band = rows[..., head]
        assert band.min() >= low and band.max() < high, f"head {head} rows outside its band"

    # bigram heads depend on 2 tokens, trigram heads on 3: changing t-2 must move only
    # the trigram half.
    a = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    b = torch.tensor([[99, 6, 7, 8]], dtype=torch.long)
    ra, rb = ported_hasher.row_indices(a), ported_hasher.row_indices(b)
    assert torch.equal(ra[0, 2, :8], rb[0, 2, :8]), "bigram heads must ignore t-2"
    assert not torch.equal(ra[0, 2, 8:], rb[0, 2, 8:]), "trigram heads must see t-2"
