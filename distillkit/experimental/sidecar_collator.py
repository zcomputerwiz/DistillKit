"""Wrap an existing collator with CPU n-gram hashing and compact GGUF row gather."""

from __future__ import annotations

import numpy as np
import torch

from distillkit.experimental.ngram_hash import NGramHasher
from distillkit.experimental.ngram_table import GGUFNGramTable


class SidecarDataCollator:
    """Preserve base collation (including teacher signals) and add ``ngram_raw``.

Hash after padding, treating masked tokens as EOS to preserve document boundaries
with either left or right padding. DataLoader pin_memory handles the returned uint8
tensor; workers never initialize CUDA. Set num_workers=0 with a resident table on
Windows. A mapped table is reopened in each spawned worker, never pickled as 28 GB
of array contents. The OS page cache is shared between those mappings.
"""

    def __init__(self, base_collator, table: GGUFNGramTable | None = None,
                 hasher: NGramHasher | None = None, shuffle_context: int = 0,
                 mode: str = "donor"):
        if mode not in ("donor", "native"):
            raise ValueError("mode must be 'donor' or 'native'")
        #: Donor collation gathers frozen GGUF bytes and emits ``ngram_raw``. Native
        #: collation stops at the row indices and emits ``ngram_ids``: the table is a
        #: model parameter, so the lookup belongs in the forward pass where it can take
        #: a gradient, not here. The mode is explicit -- a native run handed a donor
        #: batch, or the reverse, is refused by the model rather than reinterpreted.
        self.mode = mode
        if mode == "donor" and table is None:
            raise ValueError("donor collation requires a GGUF table")
        self.base_collator = base_collator
        self.table = table
        #: The matched control. A nonzero value rolls the token stream by that many
        #: positions *before* hashing, so every row is a real table row fetched for the
        #: wrong context: the gather pattern, the hit rate, the value distribution and
        #: the row norms are all preserved, and only the correspondence to this text is
        #: gone. Shuffling the retrieved bytes instead would break the IQ4_NL block
        #: structure and test dequantisation rather than the table.
        self.shuffle_context = int(shuffle_context)
        self.hasher = hasher if hasher is not None else NGramHasher()
        cfg = self.hasher.config
        if table is not None and (table.spec.head_dim != cfg.head_dim
                                  or table.spec.n_rows != self.hasher.padded_vocab_size):
            raise ValueError("table geometry does not match the n-gram hash configuration")
        self._table_factory = None

    def __call__(self, features):
        batch = dict(self.base_collator(features))
        ids = batch.get("input_ids")
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.device.type != "cpu":
            raise ValueError("base collator must return CPU input_ids with shape [batch, sequence]")
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be integer tokens")
        mask = batch.get("attention_mask")
        if mask is not None:
            if not isinstance(mask, torch.Tensor) or mask.shape != ids.shape:
                raise ValueError("sidecar collation requires a 2D, unpacked attention_mask")
            if torch.any((mask != 0) & (mask != 1)):
                raise ValueError("sidecar collation requires a binary, unpacked attention_mask")
            ids = ids.masked_fill(~mask.bool(), self.hasher.config.eos_token_id)
        if ids.numel() and (ids.min() < 0 or ids.max() >= self.hasher.config.vocab_size):
            raise ValueError("input_ids are outside the hasher's unigram vocabulary")
        if self.mode == "native":
            # `shuffle_context` still applies, and in native mode it is exactly the
            # ablation we want: real rows the model has learned, fetched for the wrong
            # n-gram. Nothing about the lookup, the value distribution or the row norms
            # changes -- only the correspondence between this text and the row identity.
            batch["ngram_ids"] = self.hasher.row_indices(
                ids.roll(self.shuffle_context, dims=-1) if self.shuffle_context else ids)
            return batch
        if self.table is None:
            self.table = GGUFNGramTable(**self._table_factory)
        rows = self.hasher.row_indices(
            ids.roll(self.shuffle_context, dims=-1) if self.shuffle_context else ids)
        raw = np.ascontiguousarray(self.table.gather_raw(rows))
        if raw.dtype != np.uint8:
            raise ValueError("table must return raw uint8 IQ4_NL rows")
        batch["ngram_raw"] = torch.from_numpy(raw)
        return batch

    def __getstate__(self):
        state = self.__dict__.copy()
        if isinstance(self.table, GGUFNGramTable):
            if self.table.resident:
                raise RuntimeError("resident GGUF tables require dataloader_num_workers=0 on Windows")
            state["table"] = None
            state["_table_factory"] = {
                "gguf_path": self.table.gguf_path,
                "tensor_name": self.table.tensor_name,
                "spec": self.table.spec,
            }
        return state
