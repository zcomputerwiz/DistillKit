"""Padding documents up to the block multiple instead of cutting them down to it.

Flooring each document to a multiple of the CSA2 block (128) cut off the final answers of
the short frontier conversations (Codex review of long round 2). With `pad_blocks` a row
keeps its whole prefix, is padded to the block multiple, and the padded positions are
masked out of every loss; the cut-down behaviour stays the default.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch" / "dense_gr"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_merged_cache import write  # noqa: E402
from teacher_kl import CachedTeacher  # noqa: E402

LENGTHS = {"a": 5, "b": 9, "c": 12}


def teacher(tmp_path, assistant_only=None):
    tokens = {doc: list(range(1, n + 1)) for doc, n in LENGTHS.items()}
    path = write(tmp_path / "cache", list(LENGTHS), tokens=tokens)
    return CachedTeacher(path, device="cpu", assistant_only=assistant_only, answer_marker=[3],
                         turn_close=7)


def test_default_still_floors(tmp_path):
    t = teacher(tmp_path)
    widths = {doc: w for group, w in t._groups(4, block=4) for doc in group}
    assert widths == {"a": 4, "b": 8, "c": 12}
    batch = t.read_batch(["b"], 8)
    assert "weight" not in batch and batch["input_ids"].shape == (1, 8)


def test_pad_keeps_every_token_and_masks_the_pads(tmp_path):
    t = teacher(tmp_path)
    t.pad_blocks = True
    widths = {doc: w for group, w in t._groups(4, block=4) for doc in group}
    assert widths == {"a": 8, "b": 12, "c": 12}
    batch = t.read_batch(["b", "c"], 12)
    ids = batch["input_ids"].numpy()
    assert ids[0, :9].tolist() == list(range(1, 10)) and ids[1].tolist() == list(range(1, 13))
    weight = batch["weight"].numpy()
    # Position t predicts t + 1: a 9-token document scores positions 0..7.
    assert weight[0].tolist() == [1.0] * 8 + [0.0] * 4
    assert weight[1].tolist() == [1.0] * 11 + [0.0]
    assert np.isfinite(batch["topk_logprobs"].numpy()).all()


def test_pad_with_assistant_only_masks_both(tmp_path):
    t = teacher(tmp_path, assistant_only=[tmp_path / "cache"])
    t.pad_blocks = True
    t._groups(4, block=4)
    batch = t.read_batch(["a"], 8)  # tokens 1..5: marker 3, then 4 5 (no close)
    assert batch["weight"][0].tolist() == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_padded_step_matches_the_unpadded_documents(tmp_path):
    from test_assistant_mask import masked_ce
    from test_dense_gr_training_step import TinyLM
    from training_step import backward_step

    t = teacher(tmp_path)
    t.pad_blocks = True
    t._groups(4, block=4)
    torch.manual_seed(0)
    model = TinyLM()
    padded = t.read_batch(["b"], 12)
    result = backward_step(model, [padded], ce=masked_ce, teacher_weight=0.5)
    assert result["targets"] == 8
    plain = {k: v[:, :9] if torch.is_tensor(v) and v.dim() > 1 else v for k, v in padded.items()
             if k != "weight"}
    model.zero_grad()
    reference = backward_step(model, [plain], ce=masked_ce, teacher_weight=0.5)
    assert reference["targets"] == 8
    # TinyLM is causal, so the pads change no real position.
    assert abs(result["loss"] - reference["loss"]) < 1e-5
    assert abs(result["teacher_kl"] - reference["teacher_kl"]) < 1e-5


def test_hedge_suppression_skips_documents_without_chat_turns(tmp_path):
    # Raw text (no turn opener 3) keeps the teacher's view whole; a chat document loses
    # the suppressed token (0, every cached top-k id here) from its answer onwards.
    tokens = {"raw": [1, 2, 4, 5, 6, 8], "chat": [1, 3, 4, 5, 6, 8]}
    path = write(tmp_path / "cache", list(tokens), tokens=tokens)
    t = CachedTeacher(path, device="cpu", answer_marker=[3], turn_close=7, min_answer_tokens=1,
                      suppress=np.array([0]))
    raw = t.read_batch(["raw"], 6)["topk_logprobs"].numpy()
    assert t.suppressed_mass == 0 and (raw == 0).all()
    t.read_batch(["chat"], 6)
    assert t.suppressed_mass > 0
