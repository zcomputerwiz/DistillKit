"""Guard against plausible-looking scores from the wrong model or targets."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from distillkit.independent_eval import (
    benchmark_record, choice_result, compare_results, complete_bypass,
    continuation_tokens, load_checkpoint, make_collator, paired_interval,
    partition, plumbing_probe, score_sequences, select_split, unseen_records,
    validate_loading,
)
from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM
from distillkit.sidecar_collator import SidecarDataCollator
from test_sidecar_collator import TinyTable, make_hasher
from test_sidecar_model import tiny_config
from distillkit.widened_residual import WidenedResidual


def test_evaluation_uses_real_collator_and_eos_padding(monkeypatch):
    hasher = make_hasher()
    table = TinyTable(hasher)
    collator = make_collator(0, table, hasher)
    calls = []
    original = SidecarDataCollator.__call__

    def spy(self, features):
        calls.append(features)
        return original(self, features)

    monkeypatch.setattr(SidecarDataCollator, "__call__", spy)
    features = [{"ids": [5, 7, 3, 12]}, {"ids": [8, 9]}]
    result = collator(features)
    expected = table.gather_raw(hasher.row_indices(torch.tensor([[5, 7, 3, 12], [8, 9, 3, 3]])))
    assert len(calls) == 1, "evaluation must execute training's collation path, not copy its hashing"
    assert np.array_equal(result["ngram_raw"].numpy(), expected), "padding must hash as EOS, not pad ID"
    assert result["input_ids"][1].tolist() == [8, 9, 0, 0], "only hashing should see substituted EOS"


def trained_model(variant="gated_residual"):
    torch.manual_seed(12)
    model = Qwen35SidecarForCausalLM(tiny_config(sidecar_variant=variant)).eval()
    with torch.no_grad():
        for p in model.model.layers[1].sidecar.parameters():
            p.add_(torch.randn_like(p) * 0.03)
    return model


def raw_collator(features):
    from test_sidecar_model import raw_batch
    batch = make_collator(0)(features)
    batch["ngram_raw"] = raw_batch(batch=len(features), length=batch["input_ids"].shape[1])
    return batch


@pytest.mark.parametrize("variant", ["gated_residual", "ple"])
def test_saved_variant_and_adapter_values_survive_evaluator_loading(tmp_path, variant):
    source = trained_model(variant)
    source.save_pretrained(tmp_path)
    loaded, audit = load_checkpoint(tmp_path)
    assert audit["variant"] == variant, "an external training YAML must not replace the checkpoint architecture"
    assert audit["adapter_exact_match"] and audit["adapter_tensor_count"] > 0
    json.dumps(audit, allow_nan=False)
    probe = plumbing_probe(loaded, raw_collator, {"ids": [5, 7, 8, 9]}, "cpu")
    assert probe["enabled_minus_bypassed_logits_max"] > 0, "a trained adapter must reach measured logits"


def test_widened_checkpoint_loads_through_its_own_class(tmp_path):
    """A widened checkpoint carries routing in every layer, and the stock class has
    nowhere to put it. Loading one with the wrong class would drop 512 trained tensors
    and report the backbone's own score as the architecture's."""
    from test_widened_residual import tiny_config as widened_config
    from distillkit.models import Qwen35WidenedForCausalLM

    torch.manual_seed(11)
    source = Qwen35WidenedForCausalLM(widened_config()).eval()
    for module in source.modules():
        if isinstance(module, WidenedResidual):
            with torch.no_grad():
                module.lambda_read.fill_(.3)
                module.write_offset.normal_(0, .1)
                module.W_up.weight.normal_(0, .05)
    ids = torch.tensor([[5, 8, 9, 3, 10, 12]])
    with torch.inference_mode():
        expected = source(input_ids=ids, use_cache=False).logits
    source.save_pretrained(tmp_path)

    loaded, audit = load_checkpoint(tmp_path)
    assert type(loaded) is Qwen35WidenedForCausalLM
    assert audit["residual_branches"] == 2
    # Two routers per layer, eight tensors each; none may be silently reinitialised.
    assert audit["adapter_tensor_count"] == 2 * source.config.num_hidden_layers * 8
    assert audit["adapter_exact_match"]
    json.dumps(audit, allow_nan=False)
    with torch.inference_mode():
        assert torch.equal(loaded(input_ids=ids, use_cache=False).logits, expected)


def test_loading_rejects_the_sketches_gr_to_ple_mismatch(tmp_path):
    trained_model().save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["sidecar_variant"] = "ple"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="did not load exactly"):
        load_checkpoint(tmp_path)


def test_missing_adapter_cannot_be_reported_as_a_successful_checkpoint(tmp_path):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    stock = Qwen3_5ForCausalLM(tiny_config())
    stock.save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["architectures"] = ["Qwen35SidecarForCausalLM"]
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="missing its adapter"):
        load_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="did not load exactly"):
        validate_loading({"missing_keys": ["model.layers.1.sidecar.W_side_proj.weight"]}, True)
    validate_loading({"unexpected_keys": ["distillation_projections.0.weight"]}, True)


def test_probe_detects_forward_that_discards_enabled_flag(monkeypatch):
    model = trained_model()
    original = model.forward

    def silently_disabled(*args, **kwargs):
        kwargs["sidecar_enabled"] = False
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", silently_disabled)
    with pytest.raises(ValueError, match="silently bypassed"):
        plumbing_probe(model, raw_collator, {"ids": [5, 7, 8, 9]}, "cpu")


def test_gr_table_ablation_is_not_complete_bypass_and_restores_on_error():
    model = trained_model()
    sidecar = model.model.layers[1].sidecar
    features = [{"ids": [5, 7, 8, 9]}]
    off = score_sequences(model, features, raw_collator, "bypassed", "cpu")
    full = score_sequences(model, features, raw_collator, "full_bypass", "cpu")
    assert off != full, "GR branches remain trained and active with the forward flag set False"
    with pytest.raises(RuntimeError):
        with complete_bypass(model, True):
            assert model.model.layers[1].sidecar is None
            raise RuntimeError("probe failure")
    assert model.model.layers[1].sidecar is sidecar, "an ablation must not leak into the next comparison"


@pytest.mark.parametrize("variant", ["gated_residual", "ple"])
def test_complete_stage1_bypass_matches_the_pre_retrofit_backbone(variant):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    model = trained_model(variant)
    config = tiny_config()
    stock = Qwen3_5ForCausalLM(config).eval()
    stock.load_state_dict({k: v for k, v in model.state_dict().items() if ".sidecar." not in k})
    features = [{"ids": [5, 7, 8, 9]}]
    reference = score_sequences(stock, features, make_collator(0), "enabled", "cpu")
    complete = score_sequences(model, features, raw_collator, "full_bypass", "cpu")
    assert complete == reference, "removing the entire stage-1 adapter must expose the unchanged backbone"
    if variant == "ple":
        assert score_sequences(model, features, raw_collator, "bypassed", "cpu") == reference


class FixedLogits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()

    def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
        logits = torch.zeros(*input_ids.shape, 10)
        # Each position predicts its successor, making a causal-shift mistake loud.
        for i in range(input_ids.shape[1] - 1):
            logits[:, i].scatter_(1, input_ids[:, i + 1, None], 5)
        return SimpleNamespace(logits=logits[:, logits_to_keep])


def test_causal_shift_continuation_mask_and_padding_counts():
    features = [{"ids": [2, 3, 4, 5], "start": 3}, {"ids": [6, 7, 8], "start": 1}]
    results = score_sequences(FixedLogits(), features, make_collator(0), "enabled", "cpu")
    expected = np.log(1 + 9 * np.exp(-5))
    assert [r["tokens"] for r in results] == [1, 2], "neither prompt nor padding may enter continuation NLL"
    assert [r["sum_nll"] for r in results] == pytest.approx([expected, 2 * expected], abs=1e-6)


def test_raw_and_normalized_choice_decisions_are_both_retained():
    result = choice_result([{"sum_nll": 2., "tokens": 1}, {"sum_nll": 3., "tokens": 3}],
                           [{"chars": 1}, {"chars": 6}], 1)
    assert result["acc"] == 0 and result["acc_token_norm"] == result["acc_char_norm"] == 1
    assert result["normalization_disagrees"], "length preference must not be concealed behind one accuracy"


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_benchmark_conventions_and_joint_boundary():
    tokenizer = CharacterTokenizer()
    mmlu = benchmark_record("mmlu", {"question": "Q?", "choices": ["a", "b", "c", "d"], "answer": 2}, tokenizer, 100)
    assert mmlu["prompt"] == "Q?\nA. a\nB. b\nC. c\nD. d\nAnswer:"
    assert mmlu["choice_text"] == list("ABCD"), "standard MMLU scores labels with the choices in context"
    arc = benchmark_record("arc", {"question": "Q?", "choices": {"label": ["1", "2"], "text": ["a", "b"]}, "answerKey": "2"}, tokenizer, 100)
    assert arc["answer"] == 1 and arc["prompt"] == "Question: Q?\nAnswer:"
    assert arc["choice_text"] == ["a", "b"], "ARC scores answer text and accepts numeric choice labels"
    class MergingTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [1, 2] if text == "prefix" else [1, 3, 4]
    with pytest.raises(ValueError, match="boundary"):
        continuation_tokens(MergingTokenizer(), "prefix", "answer")


def test_manifest_exclusion_deduplication_and_split_stability(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"documents": [{"doc_id": "used"}]}))
    source = tmp_path / "docs.jsonl"
    source.write_text("\n".join(json.dumps({"doc_id": k, "text": t}) for k, t in
                                 [("used", "old"), ("new", "new"), ("duplicate", "new")]))
    assert [r["id"] for r in unseen_records(source, [manifest])] == ["new"], "cached IDs and duplicate text cannot leak across splits"
    records = [{"id": str(i)} for i in range(100)]
    screen = select_split(records, 10, "screen")
    confirmation = select_split(records, 10, "confirmation")
    assert not ({r["id"] for r in screen} & {r["id"] for r in confirmation})
    assert screen == select_split(records[::-1], 20, "screen")[:10], "selection must not depend on source order or sample size"
    assert all(partition(r["id"]) == "confirmation" for r in confirmation)


def test_bootstrap_resamples_pairs_and_preserves_token_weighting():
    result = paired_interval([2, 30], [1, 10], denominators=[1, 10], draws=1000)
    assert result["estimate"] == pytest.approx(21 / 11), "NLL is token weighted, with documents as resampling units"
    identity = paired_interval([1, 100], [1, 100], draws=1000)
    assert identity["ci95"] == [0, 0], "unpaired resampling would fabricate uncertainty for identical checkpoints"


def test_reporting_rejects_unpaired_ids_and_partial_results():
    ref = {"checkpoint": "student", "complete": False}
    with pytest.raises(ValueError, match="incomplete"):
        compare_results(ref, ref)
    ref.update(complete=True, split="screen", tokenizer_sha256="a", task_sha256={"nll": "a"},
               records={"nll": [{"id": "one", "modes": {"enabled": {"sum_nll": 1, "tokens": 1}, "bypassed": {"sum_nll": 1, "tokens": 1}}}]})
    changed = {**ref, "task_sha256": {"nll": "b"}}
    with pytest.raises(ValueError, match="records differ"):
        compare_results(changed, ref)
