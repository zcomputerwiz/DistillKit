"""End-to-end capture -> cache -> signal doc/token alignment (gate 4, synthetic).

The single-pass capture (``sample_transformers.capture_teacher``) and the offline
signal source (``OfflineHiddenStateSignalSource``) are the two ends of the teacher
pipeline. Nothing else in the suite drives them together, so this is the regression
gate that a freshly captured cache round-trips with the right tokens at the right
positions, real top-k log probabilities, and correctly-layered anchor states — and
that a misaligned batch is rejected rather than silently feeding shifted signals.

All work is CPU-only on a tiny teacher; no GPU model load.
"""

import numpy as np
import pytest
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from distillkit.offline_cache import OfflineTeacherCache
from distillkit.sample_transformers import capture_teacher
from distillkit.signals import OfflineHiddenStateSignalSource
from test_sidecar_model import tiny_config

ANCHORS = (1, 3)
SEQ_LEN = 8
TOP_K = 4


def _capture(tmp_path, docs, **overrides):
    """Capture a list of (doc_id, tokens, split_or_None) with a seeded tiny teacher."""
    torch.manual_seed(123)
    teacher = Qwen3_5ForCausalLM(tiny_config()).eval()
    kwargs = dict(
        tokenizer_hash="ab" * 32, anchor_layers=list(ANCHORS), sequence_length=SEQ_LEN,
        top_k=TOP_K, shard_tokens=1024, eval_every=2,
    )
    kwargs.update(overrides)
    records = [
        {"doc_id": doc_id, "input_ids": list(tokens), **({"split": split} if split else {})}
        for doc_id, tokens, split in docs
    ]
    manifest = capture_teacher(teacher, records, tmp_path / "cache", **kwargs)
    return teacher, manifest


def test_capture_manifest_and_split_policy(tmp_path):
    # Two docs carry an explicit split (which must win); two fall back to the
    # eval_every=2 policy: even ordinals -> eval, odd -> train. One doc is longer
    # than sequence_length and must be truncated to exactly SEQ_LEN tokens.
    docs = [
        ("d0", list(range(5)), "train"),      # explicit train (ordinal 0 would be eval)
        ("d1", list(range(6, 13)), "eval"),   # explicit eval
        ("d2", list(range(13, 20)), None),    # fallback: ordinal 2 % 2 == 0 -> eval
        ("d3", [40 + i for i in range(12)], None),  # fallback train; length 12 > SEQ_LEN
    ]
    _capture(tmp_path, docs)
    cache = OfflineTeacherCache(tmp_path / "cache")
    assert set(cache.document_ids("train")) == {"d0", "d3"}
    assert set(cache.document_ids("eval")) == {"d1", "d2"}

    m = cache.manifest
    assert m["vocab_size"] == 64 and m["hidden_size"] == 64
    assert tuple(m["anchor_layers"]) == ANCHORS
    assert m["top_k"] == TOP_K and m["sequence_length"] == SEQ_LEN
    # The long doc was truncated to the capture window, not stored whole.
    assert len(cache.read_document("d3", tokens_only=True)["input_ids"]) == SEQ_LEN
    cache.close()


@pytest.mark.parametrize("padding", ["left", "right"])
def test_signal_source_aligns_tokens_anchors_and_topk(tmp_path, padding):
    docs = [
        ("a", [5, 8, 9, 3, 10, 12], None),
        ("b", [20, 6, 30, 44, 2, 7, 55], None),
        ("c", [1, 2, 3], None),
    ]
    teacher, _ = _capture(tmp_path, docs)
    cache = OfflineTeacherCache(tmp_path / "cache")
    source = OfflineHiddenStateSignalSource(str(tmp_path / "cache"))

    captured = {doc_id: cache.read_document(doc_id, tokens_only=True)["input_ids"].tolist()
                for doc_id, _, _ in docs}
    width = max(len(t) for t in captured.values()) + 2  # force padding on both sides' logic
    rows, masks, doc_ids = [], [], []
    spans = {}
    for doc_id, tokens, _ in docs:
        n = len(tokens)
        if padding == "right":
            row = list(tokens) + [0] * (width - n)
            mask = [1] * n + [0] * (width - n)
            start, end = 0, n
        else:
            row = [0] * (width - n) + list(tokens)
            mask = [0] * (width - n) + [1] * n
            start, end = width - n, width
        rows.append(row)
        masks.append(mask)
        doc_ids.append(doc_id)
        spans[doc_id] = (start, end)

    batch = {
        "input_ids": torch.tensor(rows, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "doc_id": doc_ids,
    }
    signal = source.get_signal(batch, return_hidden_states=True)

    b, s = len(doc_ids), width
    assert tuple(signal.sparse_ids.shape) == (b, s, TOP_K)
    assert signal.sparse_ids.dtype == torch.long
    assert tuple(signal.sparse_values.shape) == (b, s, TOP_K)
    assert signal.hidden_states is not None and len(signal.hidden_states) == len(ANCHORS)
    for anchor in signal.hidden_states:
        assert tuple(anchor.shape) == (b, s, 64) and anchor.dtype == torch.bfloat16
        assert torch.isfinite(anchor.float()).all()

    with torch.no_grad():
        for row, doc_id in enumerate(doc_ids):
            tokens = captured[doc_id]
            start, end = spans[doc_id]
            # Padded positions keep the source's sentinels (zero ids, -1e4 values).
            pad_mask = batch["attention_mask"][row].bool()
            assert torch.equal(signal.sparse_ids[row, ~pad_mask], torch.zeros(s - int(pad_mask.sum()), TOP_K, dtype=torch.long))
            assert torch.all(signal.sparse_values[row, ~pad_mask] == -1e4)

            # Independent recompute from the same teacher: proves the cache holds the
            # real top-k logprobs and correctly-layered anchors, not shifted/garbage.
            out = teacher(
                input_ids=torch.tensor([tokens]), attention_mask=torch.ones(1, len(tokens), dtype=torch.long),
                output_hidden_states=True, use_cache=False,
            )
            logits = out.logits[0].float()
            best, indices = torch.topk(logits, k=TOP_K, dim=-1)
            ref_values = (best - torch.logsumexp(logits, dim=-1, keepdim=True)).to(torch.float16)

            # Top-k ids match exactly; values match to fp16 precision.
            assert torch.equal(signal.sparse_ids[row, start:end], indices.to(torch.long))
            torch.testing.assert_close(
                signal.sparse_values[row, start:end], ref_values, atol=2e-3, rtol=0.0
            )
            # Anchor states: same fp8 cast path as capture -> bit-identical after bf16 upcast.
            for anchor_idx, layer in enumerate(ANCHORS):
                ref_hs = out.hidden_states[layer][0].to(torch.float8_e4m3fn).to(torch.bfloat16)
                assert torch.equal(signal.hidden_states[anchor_idx][row, start:end], ref_hs), (
                    f"anchor {layer} misaligned for {doc_id!r}"
                )

            # Cross-check against the raw cache read: the source copied the right doc.
            cached = cache.read_document(doc_id)
            assert np.array_equal(signal.sparse_ids[row, start:end].numpy(),
                                  cached["topk_ids"].astype(np.int64))
    cache.close()


def test_signal_source_rejects_misaligned_tokens(tmp_path):
    docs = [("a", [5, 8, 9, 3, 10, 12], None)]
    _capture(tmp_path, docs)
    source = OfflineHiddenStateSignalSource(str(tmp_path / "cache"))
    # Flip one token so the batch no longer matches what capture stored.
    bad = [5, 8, 9, 3, 10, 99]
    with pytest.raises(ValueError, match="token mismatch"):
        source.get_signal(
            {"input_ids": torch.tensor([bad]), "attention_mask": torch.ones(1, len(bad), dtype=torch.long),
             "doc_id": ["a"]},
            return_hidden_states=True,
        )
