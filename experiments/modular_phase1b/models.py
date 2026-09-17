"""Paired oracle-address models for the Phase 1b reader diagnostic."""

from __future__ import annotations

import torch
from torch import nn

from experiments.modular_phase1.language import Vocabulary
from experiments.modular_phase1.models import CausalDecoderLM, DecoderConfig, PointerReader


ADDRESS_FEATURE_WIDTH = 30


def address_only_features(
    pointers: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode only pointer presence/distance in the frozen 30-column Phase 1 schema."""
    features = torch.zeros(
        pointers.shape + (ADDRESS_FEATURE_WIDTH,), device=pointers.device, dtype=dtype
    )
    present = pointers >= 0
    positions = torch.arange(pointers.shape[1], device=pointers.device)[None, :]
    distance = (positions - pointers.clamp_min(0)).clamp_min(0)
    features[..., -2] = present.to(dtype)
    features[..., -1] = (
        distance / max(1, pointers.shape[1] - 1)
    ).to(dtype)
    return features


class DiagnosticBindingReader(PointerReader):
    """The current window reader or an exact-value selection positive control.

    Both modes have an identical state dict and integration path. ``current_window``
    retains the learned scalar attention over the seven-token declaration window.
    ``exact_value`` replaces only those attention weights with a one-hot selection of
    the observed 0/1 token in that window. The bit is found from ordinary input token
    ids; the oracle supplies only the declaration address.
    """

    MODES = ("current_window", "exact_value")

    def __init__(self, model_width: int, *, mode: str, window: int = 7) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unknown reader mode {mode!r}")
        super().__init__(ADDRESS_FEATURE_WIDTH, model_width, window=window)
        self.mode = mode

    def consumed_inputs(
        self,
        raw_embeddings: torch.Tensor,
        features: torch.Tensor,
        pointers: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        consumed = super().pointer_inputs(raw_embeddings, features, pointers)
        candidate_ids = input_ids[
            torch.arange(input_ids.shape[0], device=input_ids.device)[:, None, None],
            consumed["indices"],
        ]
        bit_mask = consumed["valid"] & (
            (candidate_ids == Vocabulary.TO_ID["0"])
            | (candidate_ids == Vocabulary.TO_ID["1"])
        )
        addressed = pointers >= 0
        bit_count = bit_mask.sum(-1)
        if torch.any(bit_count[addressed] != 1):
            raise RuntimeError("an addressed declaration window must contain exactly one bit")
        consumed["candidate_ids"] = candidate_ids
        consumed["value_mask"] = bit_mask
        if self.mode == "exact_value":
            weights = bit_mask.to(raw_embeddings.dtype)
            context = (weights.unsqueeze(-1) * consumed["candidates"]).sum(-2)
            gate = torch.sigmoid(self.pointer_gate(features))
            pointer_signal = gate * context * self.pointer_scale
            consumed["weights"] = weights
            consumed["context"] = context
            consumed["gate"] = gate
            consumed["pointer_signal"] = pointer_signal
        consumed["reader_output"] = (
            consumed["feature_signal"] + consumed["pointer_signal"]
        )
        return consumed

    def forward(
        self,
        raw_embeddings: torch.Tensor,
        features: torch.Tensor,
        pointers: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.consumed_inputs(
            raw_embeddings, features, pointers, input_ids
        )["reader_output"]


class OracleBindingDecoder(nn.Module):
    """A fresh causal decoder whose only oracle input is a declaration address."""

    def __init__(self, config: DecoderConfig, *, reader_mode: str) -> None:
        super().__init__()
        self.backbone = CausalDecoderLM(config)
        self.reader = DiagnosticBindingReader(config.width, mode=reader_mode)

    @property
    def config(self) -> DecoderConfig:
        return self.backbone.config

    def reader_inputs(
        self,
        input_ids: torch.Tensor,
        pointers: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        raw = self.backbone.embed(input_ids)
        features = address_only_features(pointers, dtype=raw.dtype)
        return self.reader.consumed_inputs(raw, features, pointers, input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        pointers: torch.Tensor,
        return_reader: bool = False,
    ):
        hidden = self.backbone.embed(input_ids)
        features = address_only_features(pointers, dtype=hidden.dtype)
        consumed = self.reader.consumed_inputs(hidden, features, pointers, input_ids)
        logits = self.backbone.decode(hidden + consumed["reader_output"], attention_mask)
        if return_reader:
            return logits, consumed
        return logits


def build_paired_model(
    *,
    seed: int,
    reader_mode: str,
    layers: int = 2,
    width: int = 64,
    heads: int = 4,
    max_sequence: int = 512,
) -> OracleBindingDecoder:
    torch.manual_seed(seed)
    config = DecoderConfig(layers, width, heads, 4 * width, max_sequence)
    return OracleBindingDecoder(config, reader_mode=reader_mode)
