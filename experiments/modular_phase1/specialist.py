"""Causal recurrent specialist plus a programmed lexical-scope table.

Learned operations
------------------
The GRU predicts the current structural event, auxiliary scope depth, and prefix validity from a
redacted prefix.  Both bit values are replaced by one ``VALUE`` id before the embedding,
so neither the hidden state nor the exported probabilities can encode a resolved value or
answer.  Only probabilities and confidence are exported; the recurrent hidden state is
private.

Programmed operations
---------------------
A streaming stack/table records declarations, exact scope depth, and emits a pointer from
a variable use to the declaration identifier token. It never stores declaration values. This explicit
operation makes binding auditable and lets the experiment ask whether a small learned
reader can use a structural address together with the backbone's still-available raw
tokens. Every update consumes only the observed prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .language import Vocabulary


EVENTS = (
    "pad", "bos", "eos", "whitespace", "open", "close", "let", "declaration",
    "assign", "value", "query", "use", "xor", "semicolon", "answer_marker", "output",
    "invalid",
)
EVENT_TO_ID = {name: index for index, name in enumerate(EVENTS)}
MAX_SCOPE_DEPTH = 8
REDACTED_VALUE_ID = len(Vocabulary.TOKENS)


@dataclass
class StructureTrace:
    events: list[int]
    depths: list[int]
    validity: list[int]
    pointers: list[int]
    alternate_pointers: list[int]
    declaration_positions: list[int]


class CausalStructureMachine:
    """Prefix-only structural analyzer with no value table and no answer computation."""

    def analyze(self, ids: Iterable[int]) -> StructureTrace:
        tokens = [Vocabulary.TOKENS[int(index)] for index in ids]
        scopes: list[dict[str, int]] = []
        events: list[int] = []
        depths: list[int] = []
        validity: list[int] = []
        pointers: list[int] = []
        alternates: list[int] = []
        declarations: list[int] = []
        state = "normal"
        pending_name: str | None = None
        pending_position = -1
        valid = True

        def active_binding(name: str) -> int:
            for scope in reversed(scopes):
                if name in scope:
                    return scope[name]
            return -1

        def alternate(correct: int) -> int:
            for scope in reversed(scopes):
                for position in reversed(tuple(scope.values())):
                    if position != correct:
                        return position
            return correct

        for position, token in enumerate(tokens):
            event = "invalid"
            pointer = -1
            other = -1
            if token == "<pad>":
                event = "pad"
            elif not valid:
                event = "invalid"
            elif token in Vocabulary.WHITESPACE:
                event = "whitespace"
            elif token == "<bos>" and position == 0:
                event = "bos"
            elif token == "<eos>":
                event = "eos"
                if state not in ("normal", "after_output") or scopes:
                    valid = False
            elif state == "after_let":
                if token in Vocabulary.VARIABLES and scopes:
                    event = "declaration"
                    pending_name = token
                    pending_position = position
                    declarations.append(position)
                    state = "after_decl"
                else:
                    valid = False
            elif state == "after_decl":
                if token == "=":
                    event = "assign"
                    state = "after_assign"
                else:
                    valid = False
            elif state == "after_assign":
                if token in ("0", "1"):
                    event = "value"
                    state = "after_decl_value"
                    # Store only the declaration address. The bit is deliberately ignored.
                    assert pending_name is not None and scopes
                    scopes[-1][pending_name] = pending_position
                else:
                    valid = False
            elif state == "after_decl_value":
                if token == ";":
                    event = "semicolon"
                    state = "normal"
                    pending_name = None
                    pending_position = -1
                else:
                    valid = False
            elif state in ("after_query", "after_xor"):
                if token in ("0", "1"):
                    event = "value"
                    state = "after_operand"
                elif token in Vocabulary.VARIABLES:
                    event = "use"
                    pointer = active_binding(token)
                    if pointer < 0:
                        valid = False
                    else:
                        other = alternate(pointer)
                        state = "after_operand"
                else:
                    valid = False
            elif state == "after_operand":
                if token == "^":
                    event = "xor"
                    state = "after_xor"
                elif token == ";":
                    event = "semicolon"
                    state = "normal"
                else:
                    valid = False
            elif state == "after_arrow":
                if token in ("0", "1"):
                    event = "output"
                    state = "after_output"
                else:
                    valid = False
            elif state == "after_output":
                valid = token == "<eos>"
                event = "eos" if valid else "invalid"
            elif token == "{":
                scopes.append({})
                event = "open"
            elif token == "}":
                if scopes:
                    scopes.pop()
                    event = "close"
                else:
                    valid = False
            elif token == "let":
                if scopes:
                    event = "let"
                    state = "after_let"
                else:
                    valid = False
            elif token == "?":
                event = "query"
                state = "after_query"
            elif token == "=>":
                event = "answer_marker"
                state = "after_arrow"
            else:
                valid = False

            if not valid:
                event = "invalid"
            events.append(EVENT_TO_ID[event])
            depths.append(min(len(scopes), MAX_SCOPE_DEPTH))
            validity.append(int(valid))
            pointers.append(pointer)
            alternates.append(other)
        return StructureTrace(events, depths, validity, pointers, alternates, declarations)

    def batch(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pointers = torch.full_like(input_ids, -1)
        alternates = torch.full_like(input_ids, -1)
        declarations = torch.zeros_like(input_ids, dtype=torch.bool)
        for row in range(input_ids.shape[0]):
            trace = self.analyze(input_ids[row].detach().cpu().tolist())
            pointers[row] = torch.tensor(trace.pointers, device=input_ids.device)
            alternates[row] = torch.tensor(trace.alternate_pointers, device=input_ids.device)
            if trace.declaration_positions:
                declarations[row, trace.declaration_positions] = True
        return pointers, alternates, declarations


def redact_ids(input_ids: torch.Tensor) -> torch.Tensor:
    result = input_ids.clone()
    bit_mask = (result == Vocabulary.TO_ID["0"]) | (result == Vocabulary.TO_ID["1"])
    result[bit_mask] = REDACTED_VALUE_ID
    return result


class RecurrentSpecialist(nn.Module):
    def __init__(self, embedding_dim: int = 24, hidden_dim: int = 48) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(len(Vocabulary.TOKENS) + 1, embedding_dim,
                                      padding_idx=Vocabulary.PAD)
        self.recurrent = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.event_head = nn.Linear(hidden_dim, len(EVENTS))
        self.depth_head = nn.Linear(hidden_dim, MAX_SCOPE_DEPTH + 1)
        self.validity_head = nn.Linear(hidden_dim, 2)

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        redacted = redact_ids(input_ids)
        hidden, _ = self.recurrent(self.embedding(redacted))
        return {
            "event_logits": self.event_head(hidden),
            "depth_logits": self.depth_head(hidden),
            "validity_logits": self.validity_head(hidden),
        }

    @property
    def learned_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def interface_width(self) -> int:
        return len(EVENTS) + MAX_SCOPE_DEPTH + 1 + 4

    def interface(
        self,
        input_ids: torch.Tensor,
        pointers: torch.Tensor,
        *,
        scope_depths: torch.Tensor | None = None,
        ablate: bool = False,
        pointer_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exported features and declaration pointers, never hidden states/values."""
        if pointer_override is not None:
            pointers = pointer_override
        if ablate:
            shape = input_ids.shape + (self.interface_width,)
            return torch.zeros(shape, device=input_ids.device), torch.full_like(pointers, -1)
        outputs = self(input_ids)
        events = outputs["event_logits"].softmax(-1)
        if scope_depths is None:
            raise ValueError("the frozen interface requires programmed causal scope depths")
        depths = F.one_hot(
            scope_depths.clamp(0, MAX_SCOPE_DEPTH), MAX_SCOPE_DEPTH + 1
        ).to(events.dtype)
        validity = outputs["validity_logits"].softmax(-1)[..., 1:2]
        confidence = events.max(-1, keepdim=True).values
        present = (pointers >= 0).unsqueeze(-1).to(events.dtype)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
        distance = (positions - pointers.clamp_min(0)).clamp_min(0)
        distance = (distance / max(1, input_ids.shape[1] - 1)).unsqueeze(-1).to(events.dtype)
        features = torch.cat((events, depths, validity, confidence, present, distance), dim=-1)
        return features, pointers

    def oracle_interface(
        self,
        input_ids: torch.Tensor,
        pointers: torch.Tensor,
        *,
        scope_depths: torch.Tensor,
        structural_events: torch.Tensor,
        structural_validity: torch.Tensor,
        pointer_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the identical exported schema from causal oracle labels.

        This is evaluation-only: it bypasses learned event/validity predictions while
        retaining the exact pointer reader and trained backbone. Values and answers are
        still absent; every oracle field is computable from the observed prefix.
        """
        if pointer_override is not None:
            pointers = pointer_override
        events = F.one_hot(structural_events, len(EVENTS)).to(torch.float32)
        depths = F.one_hot(
            scope_depths.clamp(0, MAX_SCOPE_DEPTH), MAX_SCOPE_DEPTH + 1
        ).to(torch.float32)
        validity = structural_validity.unsqueeze(-1).to(torch.float32)
        confidence = torch.ones_like(validity)
        present = (pointers >= 0).unsqueeze(-1).to(torch.float32)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
        distance = (positions - pointers.clamp_min(0)).clamp_min(0)
        distance = (distance / max(1, input_ids.shape[1] - 1)).unsqueeze(-1).to(torch.float32)
        features = torch.cat((events, depths, validity, confidence, present, distance), dim=-1)
        return features, pointers


def specialist_parameter_report(model: RecurrentSpecialist) -> dict:
    return {
        "learned_parameters": model.learned_parameters,
        "parameter_ceiling": 100_000,
        "under_ceiling": model.learned_parameters <= 100_000,
        "interface_width": model.interface_width,
        "redacted_value_id": REDACTED_VALUE_ID,
        "exports_hidden_state": False,
        "exports_values_or_answers": False,
        "learned": ["event probabilities", "auxiliary scope-depth head", "prefix validity",
                    "event confidence"],
        "programmed": ["scope stack", "exact scope depth", "name-to-declaration table",
                       "declaration pointer"],
    }
