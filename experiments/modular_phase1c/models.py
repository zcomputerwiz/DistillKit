"""Causal lookup state, width-independent readout, and output composition."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from experiments.modular_phase1.language import Vocabulary
from experiments.modular_phase1.specialist import CausalStructureMachine


BIT_IDS = (Vocabulary.TO_ID["0"], Vocabulary.TO_ID["1"])


@dataclass(frozen=True)
class LookupPrefixTrace:
    use_pointers: tuple[int, ...]
    active: tuple[bool, ...]
    binding_addresses: tuple[int, ...]
    value_addresses: tuple[int, ...]
    retrieved_values: tuple[int, ...]


def _value_address(ids: list[int], declaration: int, visible_through: int) -> int:
    """Find a declaration literal without reading beyond the observed prefix."""
    for position in range(declaration + 1, min(len(ids), visible_through + 1)):
        token = ids[position]
        if token in BIT_IDS:
            return position
        if token == Vocabulary.TO_ID[";"]:
            break
    return -1


def trace_lookup_prefix(
    ids: list[int],
    *,
    pointer_override: dict[int, int] | None = None,
) -> LookupPrefixTrace:
    """Derive lookup activation and retrieval using only each observed prefix.

    A logit position is active at ``=>`` and at whitespace following ``=>`` only when
    the completed query contained exactly one bound identifier and no XOR. The answer
    token and answer-position annotation are never consulted.
    """
    pointers = CausalStructureMachine().analyze(ids).pointers
    active = [False] * len(ids)
    bindings = [-1] * len(ids)
    values = [-1] * len(ids)
    retrieved = [-1] * len(ids)
    in_query = False
    query_complete = False
    query_xor = False
    query_pointer = -1
    query_operands = 0
    after_arrow = False
    applicable = False
    for position, token in enumerate(ids):
        if token == Vocabulary.TO_ID["?"]:
            in_query = True
            query_complete = False
            query_xor = False
            query_pointer = -1
            query_operands = 0
            after_arrow = False
            applicable = False
        elif in_query and token == Vocabulary.TO_ID["^"]:
            query_xor = True
        elif in_query and pointers[position] >= 0:
            query_operands += 1
            selected = pointers[position]
            if pointer_override is not None and position in pointer_override:
                selected = pointer_override[position]
            if query_operands == 1:
                query_pointer = selected
        elif in_query and token == Vocabulary.TO_ID[";"]:
            in_query = False
            query_complete = True
        elif token == Vocabulary.TO_ID["=>"]:
            after_arrow = True
            applicable = (
                query_complete and not query_xor and query_operands == 1
                and query_pointer >= 0
            )
        elif after_arrow and Vocabulary.TOKENS[token] not in Vocabulary.WHITESPACE:
            after_arrow = False
            applicable = False

        is_active = after_arrow and applicable and (
            token == Vocabulary.TO_ID["=>"]
            or Vocabulary.TOKENS[token] in Vocabulary.WHITESPACE
        )
        if not is_active:
            continue
        value_address = _value_address(ids, query_pointer, position)
        if value_address < 0 or ids[value_address] not in BIT_IDS:
            continue
        active[position] = True
        bindings[position] = query_pointer
        values[position] = value_address
        retrieved[position] = 0 if ids[value_address] == BIT_IDS[0] else 1
    return LookupPrefixTrace(
        tuple(pointers), tuple(active), tuple(bindings), tuple(values), tuple(retrieved)
    )


def literal_representation(value: torch.Tensor) -> torch.Tensor:
    """Two fixed features for retrieved 0/1 literals; no backbone-width dependence."""
    if torch.any((value < 0) | (value > 1)):
        raise ValueError("literal representation requires binary values")
    return torch.nn.functional.one_hot(value.long(), num_classes=2).to(torch.float32)


class DirectLookupReadout(nn.Module):
    """Six trainable parameters: a linear classifier over a two-feature literal code."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(2, 2)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        return self.classifier(representation)


def deterministic_copy_log_probs(values: torch.Tensor) -> torch.Tensor:
    """Exact one-hot conditional distribution for the retrieved observed literal."""
    result = torch.full(
        values.shape + (2,), -torch.inf, device=values.device, dtype=torch.float32
    )
    return result.scatter(-1, values.long().unsqueeze(-1), 0.0)


def compose_bit_distribution(
    backbone_log_probs: torch.Tensor,
    reader_log_probs: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Preserve total bit mass and every non-bit probability, in log space.

    Inactive rows are bitwise unchanged. Active rows implement
    ``p'(b) = (p(0) + p(1)) q_reader(b)``.
    """
    if backbone_log_probs.shape[:-1] != reader_log_probs.shape[:-1]:
        raise ValueError("reader and backbone leading dimensions differ")
    if reader_log_probs.shape[-1] != 2:
        raise ValueError("reader must provide exactly two bit log probabilities")
    if active.shape != backbone_log_probs.shape[:-1]:
        raise ValueError("activation shape does not match logits")
    if not bool(active.any()):
        return backbone_log_probs
    output = backbone_log_probs.clone()
    bit_mass = torch.logsumexp(backbone_log_probs[..., list(BIT_IDS)], dim=-1)
    replacement = bit_mass.unsqueeze(-1) + reader_log_probs
    for slot, token_id in enumerate(BIT_IDS):
        output[..., token_id] = torch.where(
            active, replacement[..., slot], backbone_log_probs[..., token_id]
        )
    return output


class LookupOutputIntegrator(nn.Module):
    """Post-hoc output integration driven only by tokens and causal pointer state."""

    def __init__(
        self,
        readout: DirectLookupReadout | None,
        *,
        deterministic_copy: bool = False,
        value_independent: bool = False,
    ) -> None:
        super().__init__()
        if deterministic_copy and readout is not None:
            raise ValueError("deterministic copy does not use a learned readout")
        if not deterministic_copy and readout is None:
            raise ValueError("a non-copy integrator requires a readout")
        self.readout = readout
        self.deterministic_copy = deterministic_copy
        self.value_independent = value_independent

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        backbone_logits: torch.Tensor,
        *,
        pointer_overrides: list[dict[int, int] | None] | None = None,
    ) -> tuple[torch.Tensor, list[LookupPrefixTrace]]:
        """Return composed log probabilities without accepting answer annotations."""
        base = torch.log_softmax(backbone_logits.float(), dim=-1)
        active = torch.zeros(input_ids.shape, dtype=torch.bool, device=input_ids.device)
        values = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)
        traces = []
        for row in range(input_ids.shape[0]):
            length = int(attention_mask[row].sum())
            override = None if pointer_overrides is None else pointer_overrides[row]
            trace = trace_lookup_prefix(
                input_ids[row, :length].tolist(), pointer_override=override
            )
            traces.append(trace)
            row_active = torch.tensor(trace.active, device=input_ids.device)
            active[row, :length] = row_active
            if row_active.any():
                retrieved = torch.tensor(trace.retrieved_values, device=input_ids.device)
                values[row, :length] = torch.where(
                    row_active, retrieved, values[row, :length]
                )
        if not bool(active.any()):
            return base, traces
        if self.deterministic_copy:
            reader_log_probs = deterministic_copy_log_probs(values)
        else:
            representation = literal_representation(values)
            if self.value_independent:
                representation = torch.zeros_like(representation)
            reader_log_probs = torch.log_softmax(self.readout(representation), dim=-1)
        return compose_bit_distribution(base, reader_log_probs, active), traces
