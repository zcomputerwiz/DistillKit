"""A student-native, trainable n-gram table behind the reference PLE block.

Every PLE arm in this project so far has read the *donor's* frozen IQ4_NL table: 51.2
billion elements captured from Qwen3.8-Flash-Next, gathered on the host and dequantised on
the way in. This is the other experiment -- the student learns its own table, jointly, with
the reference mechanism unchanged around it.

The path is the reference's, not the retrofit's::

    input_ids -> hash -> ngram_ids -> nn.Embedding -> concat 16 rows -> PLE -> stream

against the donor path's ``hash -> GGUF gather -> ngram_raw -> dequant -> PLE``. The rows
are ordinary trainable parameters of the model, so a native checkpoint is self-contained
and never needs the 28.8 GB GGUF again.

**Geometry.** The hash is untouched: ``ngram_size=3``, ``heads_per_ngram=8``, so 16 heads,
bigrams first, seed 1234, one prime address space per head just above
``ngram_vocab_size_base``, running global offsets, EOS-aware history reset. Only the two
scaled dimensions change. ``ple_embed_dim`` follows the student's hidden size, so for a
1024-wide student each of the 16 heads contributes ``1024 / 16 = 64`` dimensions and the
concatenated rows land at the stream's width with no projection. The donor's 160-wide rows
are not carried over; this table is the student's own.

**One divergence from the reference, and it is deliberate.** Flash-Next trains PLE from
scratch jointly with everything else. This student arrives pretrained, so the module sits
behind a single admission scalar::

    h' = h + rho * PLEWrite(h, E[n]),      rho_0 = 0

``rho`` is trainable and starts at exactly zero, which makes the block dormant at load and
the morphed model bit-identical to the stock one. Everything inside the block -- table,
key and value projections, norms, convolution -- initialises normally, the reference way.
That is the whole point of putting the accommodation outside: the earlier retrofit bought
its identity by zeroing ``value_proj`` and the convolution *inside* the reference
mechanism, which changed what the mechanism starts as, and (as the two-stream pilot found)
a zero inside a product is also a place gradients cannot leave.

Setting ``rho`` to zero at evaluation, or bypassing the module with
``sidecar_enabled=False``, recovers the backbone exactly -- the ON/OFF ablation needs no
second training run. The wrong-context ablation is the collator's ``shuffle_context``:
real learned rows, fetched for the wrong n-gram.
"""

from __future__ import annotations

import torch
from torch import nn

from distillkit.experimental.ngram_hash import NGramHashConfig, NGramHasher
from distillkit.experimental.ple_sidecar import PLESidecar

__all__ = ["NativePLESidecar", "native_hash_config"]


def native_hash_config(config) -> NGramHashConfig:
    """The hash geometry for a native table, from the student's own config.

    Everything except the two scaled dimensions is the reference's default and is left
    alone: changing ``seed``, ``ngram_size``, ``heads_per_ngram`` or ``ple_layer_index``
    moves every head's address range and every multiplier.
    """
    eos = getattr(config, "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        eos = eos[0]
    embed_dim = getattr(config, "sidecar_ple_embed_dim", None) or config.hidden_size
    return NGramHashConfig(
        vocab_size=config.vocab_size,
        ngram_vocab_size_base=config.sidecar_ngram_vocab_size_base,
        ple_embed_dim=embed_dim,
        eos_token_id=eos if eos is not None else NGramHashConfig.eos_token_id,
    )


def _chunked_norm(weight: torch.Tensor, rows: int = 1 << 16) -> float:
    """Frobenius norm of a large table without a full-size fp32 copy.

    ``weight.float().norm()`` on the 2.1M x 128 native table materialises roughly a
    gibibyte of temporary fp32 every time the metrics callback fires. Summing squares a
    slice at a time costs 32 MiB and gives the same number, accumulated in fp32 so the
    2.7e8 additions do not lose the tail in bf16.
    """
    total = torch.zeros((), dtype=torch.float32, device=weight.device)
    for start in range(0, weight.shape[0], rows):
        total += weight[start:start + rows].float().pow(2).sum()
    return float(total.sqrt())


class NativePLESidecar(nn.Module):
    """``h -> h + rho * PLE(h, table[ngram_ids])``, with the table owned by the model."""

    def __init__(self, config):
        super().__init__()
        hash_config = native_hash_config(config)
        # On CPU explicitly: `from_pretrained` builds the model under a meta-device
        # context, and the hasher's geometry tensors would be meta there -- reading an
        # int out of one raises. The hash is metadata, not a parameter; it does not
        # belong on the lazy path.
        with torch.device("cpu"):
            hasher = NGramHasher(hash_config)
        self.ngram_heads = hash_config.ngram_heads
        self.head_dim = hash_config.head_dim
        self.feature_dim = hash_config.ple_embed_dim
        self.ngram_vocab_size_base = hash_config.ngram_vocab_size_base
        #: The address space the hash actually produced, kept so a reloaded checkpoint can
        #: be checked against its own geometry rather than trusting the base to re-derive.
        self.head_vocab_sizes = [int(size) for size in hasher.head_vocab_sizes]
        self.head_offsets = [int(offset) for offset in hasher.head_offsets]
        self.total_vocab_size = hasher.total_vocab_size
        self.padded_vocab_size = hasher.padded_vocab_size

        # Dense, ordinary, and deliberately not `sparse=True`: this first implementation
        # changes as little as possible from the reference, so the table trains under the
        # normal optimizer. Its cost is measured rather than designed around.
        self.table = nn.Embedding(self.padded_vocab_size, self.head_dim)
        self.ple = PLESidecar(
            config.hidden_size,
            self.feature_dim,
            ngram_size=hash_config.ngram_size,
            rms_norm_eps=config.rms_norm_eps,
            identity_init=False,
        )
        # The single retrofit accommodation. Zero here, and nowhere inside the block.
        self.rho = nn.Parameter(torch.zeros(()))
        self._last_rows: tuple[int, int] | None = None

    def features(self, ngram_ids: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """``[batch, seq, heads]`` rows -> ``[batch, seq, heads * head_dim]``."""
        expected = (*hidden_states.shape[:2], self.ngram_heads)
        if ngram_ids is None:
            raise ValueError("ngram_ids is required by a native n-gram table")
        if ngram_ids.dtype not in (torch.int32, torch.int64) or tuple(ngram_ids.shape) != expected:
            raise ValueError(f"ngram_ids must be integer rows with shape {expected}")
        rows = ngram_ids.to(device=self.table.weight.device, dtype=torch.long)
        if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= self.padded_vocab_size):
            raise ValueError("ngram_ids outside the native table's address space")
        if self.training:
            # Two cheap scalars off a tensor already on the device; no whole-table scan.
            self._last_rows = (int(torch.unique(rows).numel()), int(rows.numel()))
        return self.table(rows).flatten(-2).to(hidden_states.dtype)

    def forward(self, hidden_states, ngram_ids=None, sidecar_enabled=True):
        if not sidecar_enabled:
            return hidden_states
        features = self.features(ngram_ids, hidden_states)
        write = self.ple.write(hidden_states.to(features.device), features)
        return hidden_states + (self.rho.to(write.dtype) * write).to(hidden_states.device)

    @torch.no_grad()
    def gate_report(self, prefix: str = "native_ple") -> dict:
        """Just enough to tell whether the mechanism is alive.

        ``rho`` leaving zero is the first thing to happen and the table norm moving is the
        second; a gate that returns one number for every token is a constant admission
        rather than a selector, which is what ``gate_std`` separates. The row counters are
        per-batch, not per-table: a whole-table scan every step is the kind of
        instrumentation this task is explicitly not building yet.
        """
        report = {
            f"{prefix}/rho": self.rho.float().item(),
            f"{prefix}/table_weight_norm": _chunked_norm(self.table.weight),
            f"{prefix}/value_proj_norm": self.ple.value_proj.weight.float().norm().item(),
            f"{prefix}/key_proj_norm": self.ple.key_proj.weight.float().norm().item(),
            f"{prefix}/conv_norm": self.ple.conv1d.weight.float().norm().item(),
        }
        stats = self.ple._last_gate_stats
        if stats is not None:
            mean, std, _, _ = stats.tolist()
            report[f"{prefix}/gate_mean"] = mean
            report[f"{prefix}/gate_std"] = std
        if self._last_rows is not None:
            unique, touched = self._last_rows
            report[f"{prefix}/unique_rows_in_batch"] = float(unique)
            report[f"{prefix}/row_touch_fraction"] = unique / max(self.padded_vocab_size, 1)
        return report

    def geometry(self) -> dict:
        """What the hash produced, for the report and for checkpoint validation."""
        return {
            "ngram_vocab_size_base": self.ngram_vocab_size_base,
            "ngram_heads": self.ngram_heads,
            "head_dim": self.head_dim,
            "feature_dim": self.feature_dim,
            "head_vocab_sizes": self.head_vocab_sizes,
            "head_offsets": self.head_offsets,
            "total_vocab_size": self.total_vocab_size,
            "padded_vocab_size": self.padded_vocab_size,
            "table_parameters": self.padded_vocab_size * self.head_dim,
        }
