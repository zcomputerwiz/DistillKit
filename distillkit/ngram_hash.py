"""Bit-exact port of the Qwen4-Exp (Qwen3.8-Flash-Next) hashed n-gram row indexer.

This is a **verbatim port** of the index-computing half of
``transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextNGramEmbedding``
(and the module-level helpers it depends on). It is vendored rather than imported
so that the training hot path does not depend on ``qwen4_exp`` internals staying
stable across transformers releases -- but ``tests/test_ngram_hash.py`` asserts
bit-identity against the live reference on every run, so drift is loud.

Do not "clean up" the arithmetic here. Every operation is load-bearing:

* ``token_id * multiplier`` is a signed int64 multiply that runs close to the int64
  ceiling. ``build_layer_multipliers`` bounds every multiplier by
  ``(2**63 - 1) // vocab_size``, so the product cannot wrap *and* its sign bit is
  always clear. That invariant is what keeps the XOR fold -- and therefore every row
  index -- non-negative. Widen the vocab or loosen the bound and rows silently go
  negative. ``tests/test_ngram_hash.py`` pins it.
* Row indices are global: per-head vocab sizes are 16 *distinct primes* just above
  ``ngram_vocab_size_base``, and each head's rows are offset by the running sum of
  all preceding head vocab sizes.

For Qwen3.8-Flash-Next (``ngram_size=3``, ``heads_per_ngram=8``, one PLE layer) this
yields 16 heads of 160 dims each -> 2560, over a table of 320,001,536 rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

__all__ = [
    "NGramHashConfig",
    "NGramHasher",
    "FLASH_NEXT_NGRAM_CONFIG",
    "splitmix64",
    "build_layer_multipliers",
    "find_nth_prime_after",
]

# --- verbatim from modeling_qwen4_exp.py -------------------------------------

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def build_layer_multipliers(
    unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int
) -> torch.Tensor:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (splitmix64(value) % half_bound) + 1)
    return torch.tensor(multipliers, dtype=torch.long)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


# --- configuration -----------------------------------------------------------


@dataclass(frozen=True)
class NGramHashConfig:
    """Every constant the row indexer needs, all sourced from the Flash-Next config.

    ``ple_layer_index`` is the *position of this PLE layer within* ``ple_layer_ids``
    (not the decoder layer index). Flash-Next has ``ple_layer_ids=[2]``, so the single
    PLE layer has ``ple_layer_index=0``. Getting this wrong shifts every head's global
    index and therefore every row offset and multiplier.
    """

    vocab_size: int = 248320
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    seed: int = 1234
    ple_layer_index: int = 0
    ple_embed_dim: int = 2560
    eos_token_id: int = 248044

    @property
    def context_len(self) -> int:
        return self.ngram_size - 1

    @property
    def ngram_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def head_dim(self) -> int:
        return self.ple_embed_dim // self.ngram_heads

    @classmethod
    def from_pretrained_config(cls, config, ple_layer_index: int = 0) -> "NGramHashConfig":
        """Build from a ``Qwen4ExpTextConfig`` (or the ``text_config`` of the VLM)."""
        text = getattr(config, "text_config", config)
        eos = text.eos_token_id
        if isinstance(eos, (list, tuple)):
            eos = eos[0]
        return cls(
            vocab_size=text.vocab_size,
            ngram_size=text.ngram_size,
            heads_per_ngram=text.heads_per_ngram,
            ngram_vocab_size_base=text.ngram_vocab_size_base,
            make_ngram_vocab_size_divisible_by=text.make_ngram_vocab_size_divisible_by,
            seed=text.seed,
            ple_layer_index=ple_layer_index,
            ple_embed_dim=text.ple_embed_dim,
            eos_token_id=eos,
        )


FLASH_NEXT_NGRAM_CONFIG = NGramHashConfig()


# --- the hasher --------------------------------------------------------------


@dataclass
class NGramHasher:
    """Turns ``input_ids`` into global n-gram table row indices, bit-exactly.

    This deliberately does **not** own an embedding table -- it only produces
    ``[batch, seq, ngram_heads]`` int64 rows. The 51.2 GB table is fetched
    elsewhere (memmap gather on the host), which is the whole point of splitting
    the index computation out: it can run in the collator, off the GPU critical path.
    """

    config: NGramHashConfig = field(default_factory=NGramHashConfig)

    head_vocab_sizes: torch.Tensor = field(init=False, repr=False)
    head_offsets: torch.Tensor = field(init=False, repr=False)
    layer_multipliers: torch.Tensor = field(init=False, repr=False)
    total_vocab_size: int = field(init=False)
    padded_vocab_size: int = field(init=False)

    def __post_init__(self) -> None:
        cfg = self.config
        sizes: list[int] = []
        offsets: list[int] = []
        total = 0
        for head_idx in range(cfg.ngram_heads):
            global_head_idx = cfg.ple_layer_index * cfg.ngram_heads + head_idx
            size = find_nth_prime_after(cfg.ngram_vocab_size_base - 1, global_head_idx + 1)
            sizes.append(size)
            offsets.append(total)
            total += size

        self.head_vocab_sizes = torch.tensor(sizes, dtype=torch.long)
        self.head_offsets = torch.tensor(offsets, dtype=torch.long)
        self.layer_multipliers = build_layer_multipliers(
            cfg.vocab_size, cfg.ngram_size, cfg.ple_layer_index, cfg.seed
        )
        self.total_vocab_size = total
        divisor = cfg.make_ngram_vocab_size_divisible_by
        self.padded_vocab_size = math.ceil(total / divisor) * divisor

    # -- verbatim from Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos ------

    def _shift_right_ignore_eos(self, token_ids: torch.Tensor, shift: int) -> torch.Tensor:
        if shift == 0:
            return token_ids
        eos_token_id = self.config.eos_token_id
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        eos_positions = torch.where(token_ids == eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1
        )
        segment_start = previous_eos + 1
        position_in_segment = positions.unsqueeze(0) - segment_start
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        shifted = token_ids.gather(dim=1, index=gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), eos_token_id))

    # -- index computation ----------------------------------------------------

    def row_indices(
        self,
        input_ids: torch.Tensor,
        previous_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Global table rows for every token.

        Args:
            input_ids: ``[batch, seq]`` int64 token ids.
            previous_context: ``[batch, context_len]`` tokens preceding ``input_ids``.
                ``None`` means "start of sequence", which the reference fills with EOS --
                the correct choice for teacher-forced training over independent documents.

        Returns:
            ``[batch, seq, ngram_heads]`` int64 rows into the padded n-gram table.
            Head order is the reference's: bigram heads first (``0..heads_per_ngram-1``),
            then trigram heads, matching the 160-dim slices of the concatenated 2560 vector.
        """
        cfg = self.config
        input_ids = input_ids.long()
        device = input_ids.device

        if previous_context is None:
            previous_context = input_ids.new_full(
                (input_ids.shape[0], cfg.context_len), cfg.eos_token_id
            )

        token_history = torch.cat([previous_context, input_ids], dim=-1)
        shifted_tokens = [
            self._shift_right_ignore_eos(token_history, shift) for shift in range(cfg.ngram_size)
        ]

        multipliers = self.layer_multipliers.to(device)
        vocab_sizes = self.head_vocab_sizes.to(device)
        offsets = self.head_offsets.to(device)

        blocks = []
        for ngram in range(2, cfg.ngram_size + 1):
            start_idx = (ngram - 2) * cfg.heads_per_ngram
            end_idx = start_idx + cfg.heads_per_ngram
            mixed_ids = shifted_tokens[0] * multipliers[0]
            for position in range(1, ngram):
                mixed_ids = torch.bitwise_xor(
                    mixed_ids, shifted_tokens[position] * multipliers[position]
                )
            head_vocab_sizes = vocab_sizes[start_idx:end_idx]
            head_offsets = offsets[start_idx:end_idx]
            # mixed_ids is non-negative by the multiplier bound (see module docstring),
            # so remainder and fmod agree here -- but keep remainder to match the
            # reference exactly, and to stay correct if that bound ever changes.
            ngram_ids = torch.remainder(mixed_ids.unsqueeze(-1), head_vocab_sizes.view(1, 1, -1))
            blocks.append(ngram_ids + head_offsets.view(1, 1, -1))

        return torch.cat(blocks, dim=-1)[:, -input_ids.shape[1] :]
