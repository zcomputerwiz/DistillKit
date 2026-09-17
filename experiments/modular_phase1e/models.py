"""Scalar category-admission corrections for the frozen Phase 1d distribution."""

from __future__ import annotations

import math

import torch
from torch import nn

from experiments.modular_phase1.language import Vocabulary
from experiments.modular_phase1.specialist import EVENTS, MAX_SCOPE_DEPTH
from experiments.modular_phase1c.models import BIT_IDS


CAUSAL_POSITION_SCALE = 512.0


FEATURE_NAMES = tuple([f"event_probability_{index}" for index in range(len(EVENTS))] + [
    "validity_probability",
    "event_confidence",
    "programmed_scope_depth",
    "current_is_whitespace",
    "current_is_answer_marker",
    "fixed_scale_causal_position",
    "backbone_bit_probability",
    "backbone_bit_log_odds",
    "backbone_entropy",
])


class ConstantAdmission(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.correction = nn.Parameter(torch.zeros(()))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.correction.expand(features.shape[:-1])


class ContextualAdmission(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.affine = nn.Linear(len(FEATURE_NAMES), 1)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.affine(features).squeeze(-1)


def apply_category_admission(
    phase1d_log_probs: torch.Tensor,
    correction: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Add one correction to both bits and preserve within-category conditionals."""
    if correction.shape != phase1d_log_probs.shape[:-1] or active.shape != correction.shape:
        raise ValueError("admission shapes do not align")
    if not bool(active.any()):
        return phase1d_log_probs
    output = phase1d_log_probs.clone()
    selected = phase1d_log_probs[active]
    delta = correction[active]
    adjusted = selected.clone()
    adjusted[:, list(BIT_IDS)] = adjusted[:, list(BIT_IDS)] + delta.unsqueeze(-1)
    # Subtract the change in log normalizer. At delta=0 the two reductions are
    # bitwise identical, so zero initialization agrees exactly while retaining gradient.
    normalization_change = torch.logsumexp(adjusted, -1) - torch.logsumexp(selected, -1)
    adjusted = adjusted - normalization_change.unsqueeze(-1)
    output[active] = adjusted
    return output


def build_contextual_features(
    input_ids: torch.Tensor,
    backbone_log_probs: torch.Tensor,
    event_logits: torch.Tensor,
    validity_logits: torch.Tensor,
    programmed_depths: torch.Tensor,
) -> torch.Tensor:
    """Frozen causal features. Retrieved value identity is deliberately absent."""
    event_probability = event_logits.float().softmax(-1)
    validity = validity_logits.float().softmax(-1)[..., 1:2]
    confidence = event_probability.max(-1, keepdim=True).values
    depth = (programmed_depths.float() / MAX_SCOPE_DEPTH).unsqueeze(-1)
    whitespace = torch.zeros_like(input_ids, dtype=torch.float32)
    for token in Vocabulary.WHITESPACE:
        whitespace += (input_ids == Vocabulary.TO_ID[token]).float()
    marker = (input_ids == Vocabulary.TO_ID["=>"]).float()
    # The scale is a frozen property of the 512-token backbone, not the current
    # tensor width.  Consequently a prefix receives the same feature when it is
    # truncated, given another unseen suffix, or padded beside a longer example.
    position = torch.arange(input_ids.shape[1], device=input_ids.device).float()
    position = (position / CAUSAL_POSITION_SCALE)[None].expand_as(input_ids)
    bit_log_mass = torch.logsumexp(backbone_log_probs[..., list(BIT_IDS)], -1)
    nonbit_ids = [index for index in range(len(Vocabulary.TOKENS)) if index not in BIT_IDS]
    nonbit_log_mass = torch.logsumexp(backbone_log_probs[..., nonbit_ids], -1)
    bit_probability = bit_log_mass.exp()
    bit_log_odds = bit_log_mass - nonbit_log_mass
    probability = backbone_log_probs.exp()
    entropy = -(probability * backbone_log_probs).sum(-1) / math.log(len(Vocabulary.TOKENS))
    scalar = torch.stack((
        whitespace, marker, position, bit_probability, bit_log_odds, entropy,
    ), dim=-1)
    return torch.cat((event_probability, validity, confidence, depth, scalar), dim=-1)
