"""Correctness checks for learned-specialist handoff boundaries."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from experiments.modular_phase1.language import Vocabulary, tokenize
from experiments.modular_phase1.specialist import EVENTS, CausalStructureMachine
from experiments.modular_phase1d.deployment import deployed_lookup_trace


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "experiments" / "modular_phase1d" / "configs" / "diagnostic.json"


class _PerfectEventModel(torch.nn.Module):
    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = torch.full(
            input_ids.shape + (len(EVENTS),), -100.0, device=input_ids.device
        )
        for row in range(input_ids.shape[0]):
            events = CausalStructureMachine().analyze(input_ids[row].tolist()).events
            for position, event in enumerate(events):
                logits[row, position, event] = 100.0
        return {"event_logits": logits}


class _PostArrowWhitespaceInvalid(_PerfectEventModel):
    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        result = super().forward(input_ids)
        logits = result["event_logits"]
        for row in range(input_ids.shape[0]):
            for position in range(1, input_ids.shape[1]):
                if (
                    input_ids[row, position - 1] == Vocabulary.TO_ID["=>"]
                    and Vocabulary.TOKENS[int(input_ids[row, position])]
                    in Vocabulary.WHITESPACE
                ):
                    logits[row, position] = -100.0
                    logits[row, position, EVENTS.index("invalid")] = 100.0
        return result


def _score(ids: list[int]) -> int:
    return len(ids) - 3


def test_deployed_path_uses_learned_events_and_programmed_pointer_content():
    ids = tokenize("{let a=0;let b=1;?a;}=> 0")
    device = torch.device("cpu")
    trace = deployed_lookup_trace(_PerfectEventModel(), ids, device)
    score = _score(ids)
    assert trace.activated[score]
    assert trace.active[score]
    assert trace.retrieved_values[score] == 0
    use = max(position for position, pointer in enumerate(trace.programmed_pointers)
              if pointer >= 0)
    declaration_b = ids.index(Vocabulary.TO_ID["b"])
    changed = deployed_lookup_trace(
        _PerfectEventModel(), ids, device,
        pointer_override={use: declaration_b},
    )
    assert changed.retrieved_values[score] == 1
    assert changed.value_addresses[score] < score


def test_deployed_path_is_prefix_causal_and_excludes_xor():
    model = _PerfectEventModel()
    ids = tokenize("{let a=1;?a;}=> 1")
    full = deployed_lookup_trace(model, ids, torch.device("cpu"))
    for length in range(1, len(ids) + 1):
        prefix = deployed_lookup_trace(model, ids[:length], torch.device("cpu"))
        assert prefix.predicted_events == full.predicted_events[:length]
        assert prefix.active == full.active[:length]
        assert prefix.retrieved_values == full.retrieved_values[:length]
    xor = tokenize("{let a=1;let b=0;?a^b;}=>1")
    assert not any(deployed_lookup_trace(model, xor, torch.device("cpu")).activated)
    signature = inspect.signature(deployed_lookup_trace)
    assert "answer_position" not in signature.parameters
    assert "target" not in signature.parameters


def test_raw_post_arrow_whitespace_preserves_learned_marker_activation():
    ids = tokenize("{let a=1;?a;}=> 1")
    trace = deployed_lookup_trace(
        _PostArrowWhitespaceInvalid(), ids, torch.device("cpu")
    )
    score = _score(ids)
    assert EVENTS[trace.predicted_events[score]] == "invalid"
    assert trace.active[score]
    assert trace.retrieved_values[score] == 1


def test_phase1d_is_strict_bounded_and_freezes_phase1c_path():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["qualification"]["binding_accuracy"] == 0.99
    assert config["qualification"]["structural_event_accuracy"] == 0.99
    assert config["specialist"]["max_parameters"] <= 100_000
    assert config["budget"]["specialist_tokens"] <= 2_097_152
    assert config["budget"]["specialist_accelerator_seconds"] <= 1800
    assert config["readout_checkpoint"] == "scratch/modular-phase1c/readout.pt"
    assert config["base_checkpoint"].endswith(
        "backbones/l6_w256/plain/seed_11/checkpoint.pt"
    )
