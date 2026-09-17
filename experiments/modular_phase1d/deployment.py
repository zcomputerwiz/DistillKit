"""Deploy the learned event model with its programmed causal binding state."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from experiments.modular_phase1.language import Vocabulary
from experiments.modular_phase1.specialist import (
    EVENTS,
    CausalStructureMachine,
    RecurrentSpecialist,
)
from experiments.modular_phase1c.models import BIT_IDS


@dataclass(frozen=True)
class DeployedLookupTrace:
    predicted_events: tuple[int, ...]
    programmed_pointers: tuple[int, ...]
    activated: tuple[bool, ...]
    active: tuple[bool, ...]
    binding_addresses: tuple[int, ...]
    value_addresses: tuple[int, ...]
    retrieved_values: tuple[int, ...]


def _value_address(ids: list[int], declaration: int, visible_through: int) -> int:
    for position in range(declaration + 1, min(len(ids), visible_through + 1)):
        if ids[position] in BIT_IDS:
            return position
        if ids[position] == Vocabulary.TO_ID[";"]:
            break
    return -1


@torch.inference_mode()
def deployed_lookup_trace(
    model: RecurrentSpecialist,
    ids: list[int],
    device: torch.device,
    *,
    pointer_override: dict[int, int] | None = None,
) -> DeployedLookupTrace:
    """Use learned events and the specialist's programmed table on the observed prefix."""
    tensor = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    predicted = model(tensor)["event_logits"].argmax(-1)[0].cpu().tolist()
    programmed = CausalStructureMachine().analyze(ids).pointers
    activated = [False] * len(ids)
    active = [False] * len(ids)
    bindings = [-1] * len(ids)
    value_addresses = [-1] * len(ids)
    retrieved = [-1] * len(ids)
    in_query = False
    complete = False
    xor = False
    pointer = -1
    use_position = -1
    operands = 0
    after_marker = False
    applicable = False
    for position, event_id in enumerate(predicted):
        event = EVENTS[event_id]
        lexical_whitespace = Vocabulary.TOKENS[ids[position]] in Vocabulary.WHITESPACE
        if event == "query":
            in_query = True
            complete = False
            xor = False
            pointer = -1
            use_position = -1
            operands = 0
            after_marker = False
            applicable = False
        elif in_query and event == "xor":
            xor = True
        elif in_query and event == "use":
            selected = programmed[position]
            if pointer_override is not None and position in pointer_override:
                selected = pointer_override[position]
            operands += 1
            if operands == 1:
                pointer = selected
                use_position = position
        elif in_query and event == "value":
            operands += 1
        elif in_query and event == "semicolon":
            in_query = False
            complete = True
        elif event == "answer_marker":
            after_marker = True
            applicable = complete and not xor and operands == 1 and pointer >= 0
        elif after_marker and not lexical_whitespace:
            after_marker = False
            applicable = False

        # The learned event opens applicability at the marker. Carrying it across
        # literal whitespace is deterministic lexical preprocessing, exactly as in
        # Phase 1c, and does not substitute an oracle structural label.
        is_active = after_marker and applicable and (
            event == "answer_marker" or lexical_whitespace
        )
        if not is_active:
            continue
        activated[position] = True
        source = _value_address(ids, pointer, position)
        if source < 0 or ids[source] not in BIT_IDS:
            continue
        active[position] = True
        bindings[position] = pointer
        value_addresses[position] = source
        retrieved[position] = int(ids[source] == BIT_IDS[1])
        if use_position < 0:
            raise AssertionError("active lookup has no learned use event")
    return DeployedLookupTrace(
        tuple(predicted), tuple(programmed), tuple(activated), tuple(active), tuple(bindings),
        tuple(value_addresses), tuple(retrieved),
    )
