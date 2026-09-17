"""Tiny causal decoders, frozen-specialist composition, and matched controls."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .language import Vocabulary
from .specialist import RecurrentSpecialist


@dataclass(frozen=True)
class DecoderConfig:
    layers: int
    width: int
    heads: int
    feedforward: int
    max_sequence: int
    dropout: float = 0.0


class CausalBlock(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width)
        self.attention = nn.MultiheadAttention(
            config.width, config.heads, dropout=config.dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(config.width)
        self.feedforward = nn.Sequential(
            nn.Linear(config.width, config.feedforward),
            nn.GELU(),
            nn.Linear(config.feedforward, config.width),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        length = hidden.shape[1]
        causal = torch.ones((length, length), dtype=torch.bool, device=hidden.device).triu(1)
        normalized = self.norm1(hidden)
        attended = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        hidden = hidden + self.dropout(attended)
        hidden = hidden + self.dropout(self.feedforward(self.norm2(hidden)))
        return hidden


class CausalDecoderLM(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            len(Vocabulary.TOKENS), config.width, padding_idx=Vocabulary.PAD,
        )
        self.position_embedding = nn.Embedding(config.max_sequence, config.width)
        self.blocks = nn.ModuleList(CausalBlock(config) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(config.width)
        self.lm_head = nn.Linear(config.width, len(Vocabulary.TOKENS), bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[1] > self.config.max_sequence:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds {self.config.max_sequence}"
            )
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        return self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]

    def decode(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        padding = None if attention_mask is None else ~attention_mask.bool()
        for block in self.blocks:
            hidden = block(hidden, padding)
        return self.lm_head(self.final_norm(hidden))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        return self.decode(self.embed(input_ids), attention_mask)


class PointerReader(nn.Module):
    """A small reader that combines exported structure with raw-token pointer windows."""

    def __init__(self, feature_width: int, model_width: int, window: int = 7) -> None:
        super().__init__()
        self.feature_width = feature_width
        self.model_width = model_width
        self.window = window
        self.feature_projection = nn.Linear(feature_width, model_width)
        self.pointer_score = nn.Linear(model_width, 1, bias=False)
        self.pointer_gate = nn.Linear(feature_width, 1)
        self.pointer_scale = nn.Parameter(torch.zeros(model_width))
        # Exact paired initialization: before training, composition equals the plain arm.
        nn.init.zeros_(self.feature_projection.weight)
        nn.init.zeros_(self.feature_projection.bias)
        nn.init.zeros_(self.pointer_gate.weight)
        nn.init.zeros_(self.pointer_gate.bias)

    def pointer_inputs(
        self,
        raw_embeddings: torch.Tensor,
        features: torch.Tensor,
        pointers: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Expose the exact gathered tensors used by the trained reader for auditing."""
        batch, length, _ = raw_embeddings.shape
        offsets = torch.arange(self.window, device=raw_embeddings.device)
        indices = pointers.clamp_min(0).unsqueeze(-1) + offsets
        positions = torch.arange(length, device=raw_embeddings.device)[None, :, None]
        valid = (pointers >= 0).unsqueeze(-1) & (indices < length) & (indices <= positions)
        indices = indices.clamp(0, length - 1)
        batch_indices = torch.arange(batch, device=raw_embeddings.device)[:, None, None]
        candidates = raw_embeddings[batch_indices, indices]
        scores = self.pointer_score(candidates).squeeze(-1).masked_fill(~valid, -1e4)
        weights = scores.softmax(-1) * valid.to(scores.dtype)
        context = (weights.unsqueeze(-1) * candidates).sum(-2)
        gate = torch.sigmoid(self.pointer_gate(features))
        pointer_signal = gate * context * self.pointer_scale
        return {
            "indices": indices,
            "valid": valid,
            "candidates": candidates,
            "scores": scores,
            "weights": weights,
            "context": context,
            "gate": gate,
            "pointer_signal": pointer_signal,
            "feature_signal": self.feature_projection(features),
        }

    def forward(
        self,
        raw_embeddings: torch.Tensor,
        features: torch.Tensor,
        pointers: torch.Tensor,
    ) -> torch.Tensor:
        consumed = self.pointer_inputs(raw_embeddings, features, pointers)
        return consumed["feature_signal"] + consumed["pointer_signal"]


class SpecialistDecoderLM(nn.Module):
    def __init__(self, backbone: CausalDecoderLM, specialist: RecurrentSpecialist) -> None:
        super().__init__()
        self.backbone = backbone
        self.specialist = specialist
        self.specialist.requires_grad_(False)
        self.specialist.eval()
        self.reader = PointerReader(specialist.interface_width, backbone.config.width)

    @property
    def config(self) -> DecoderConfig:
        return self.backbone.config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        pointers: torch.Tensor,
        scope_depths: torch.Tensor,
        alternate_pointers: torch.Tensor | None = None,
        structural_events: torch.Tensor | None = None,
        structural_validity: torch.Tensor | None = None,
        pointer_override: torch.Tensor | None = None,
        intervention: str = "none",
    ) -> torch.Tensor:
        self.specialist.eval()
        ablate = intervention == "ablate"
        override = alternate_pointers if intervention == "wrong_pointer" else pointer_override
        with torch.no_grad():
            if intervention == "oracle":
                if structural_events is None or structural_validity is None:
                    raise ValueError("oracle intervention requires event and validity labels")
                features, selected = self.specialist.oracle_interface(
                    input_ids, pointers, scope_depths=scope_depths,
                    structural_events=structural_events,
                    structural_validity=structural_validity,
                    pointer_override=override,
                )
            else:
                features, selected = self.specialist.interface(
                    input_ids, pointers, scope_depths=scope_depths, ablate=ablate,
                    pointer_override=override,
                )
        hidden = self.backbone.embed(input_ids)
        hidden = hidden + self.reader(hidden, features.to(hidden.dtype), selected)
        return self.backbone.decode(hidden, attention_mask)


def learned_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _copy_paired_state(base: CausalDecoderLM, matched: CausalDecoderLM) -> None:
    """Copy every common tensor and overlapping FFN slice from the paired base init."""
    source = base.state_dict()
    target = matched.state_dict()
    with torch.no_grad():
        for name, value in target.items():
            if name not in source:
                continue
            origin = source[name]
            if origin.shape == value.shape:
                value.copy_(origin)
                continue
            if len(origin.shape) != len(value.shape):
                continue
            slices = tuple(slice(0, min(left, right)) for left, right in zip(origin.shape, value.shape))
            value[slices].copy_(origin[slices])
        matched.load_state_dict(target)


def build_models(
    *,
    layers: int,
    width: int,
    heads: int,
    max_sequence: int,
    seed: int,
    specialist: RecurrentSpecialist,
) -> dict[str, nn.Module]:
    """Construct all three paired arms and solve the matched-control FFN width exactly."""
    base_config = DecoderConfig(layers, width, heads, 4 * width, max_sequence)
    torch.manual_seed(seed)
    plain = CausalDecoderLM(base_config)
    torch.manual_seed(seed)
    specialist_backbone = CausalDecoderLM(base_config)
    composed = SpecialistDecoderLM(specialist_backbone, specialist)
    extra = learned_parameter_count(specialist) + learned_parameter_count(composed.reader)
    # Each extra FFN unit contributes width weights in, width weights out, and one bias
    # per layer. Spread the total specialist+reader footprint across all layers.
    per_unit = layers * (2 * width + 1)
    extra_units = max(1, math.ceil(extra / per_unit))
    matched_config = DecoderConfig(layers, width, heads, 4 * width + extra_units, max_sequence)
    torch.manual_seed(seed + 1_000_000)
    matched = CausalDecoderLM(matched_config)
    _copy_paired_state(plain, matched)
    return {"plain": plain, "specialist": composed, "matched": matched}


def model_parameter_report(models: dict[str, nn.Module]) -> dict:
    report = {}
    for name, model in models.items():
        config = model.config if not isinstance(model, SpecialistDecoderLM) else model.backbone.config
        report[name] = {
            "total_parameters": learned_parameter_count(model),
            "trainable_parameters": trainable_parameter_count(model),
            "config": asdict(config),
        }
    report["matched_total_error"] = (
        report["matched"]["total_parameters"] - report["specialist"]["total_parameters"]
    )
    report["matched_relative_error"] = abs(report["matched_total_error"]) / report[
        "specialist"
    ]["total_parameters"]
    return report
