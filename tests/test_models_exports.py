"""Verify export parity, backward compatibility, and model registry resolution."""

from types import SimpleNamespace
import pytest

import distillkit.models as models
import distillkit.models.registry as registry
import distillkit.models.qwen35 as qwen35

import distillkit.convert_gguf_student as legacy_convert_gguf
import distillkit.linear_attention_dispatch as legacy_linear_attn
import distillkit.gqa_dispatch as legacy_gqa
import distillkit.models.qwen35_sidecar as legacy_sidecar
import distillkit.models.qwen35_widened as legacy_widened

import distillkit.models.qwen35.convert_gguf as qwen35_convert_gguf
import distillkit.models.qwen35.linear_attention_dispatch as qwen35_linear_attn
import distillkit.models.qwen35.gqa_dispatch as qwen35_gqa
import distillkit.models.qwen35.sidecar as qwen35_sidecar
import distillkit.models.qwen35.widened as qwen35_widened


def test_qwen35_exports_parity():
    # Architectures
    assert qwen35.Qwen35SidecarForCausalLM is qwen35_sidecar.Qwen35SidecarForCausalLM is legacy_sidecar.Qwen35SidecarForCausalLM is models.Qwen35SidecarForCausalLM
    assert qwen35.Qwen35WidenedForCausalLM is qwen35_widened.Qwen35WidenedForCausalLM is legacy_widened.Qwen35WidenedForCausalLM is models.Qwen35WidenedForCausalLM

    # Conversion
    assert qwen35.convert_gguf_to_hf is qwen35_convert_gguf.convert_gguf_to_hf is legacy_convert_gguf.convert_gguf_to_hf
    assert qwen35.derive_text_config is qwen35_convert_gguf.derive_text_config is legacy_convert_gguf.derive_text_config
    assert qwen35.expected_key_set is qwen35_convert_gguf.expected_key_set is legacy_convert_gguf.expected_key_set

    # Dispatches
    assert qwen35.install_device_aware_linear_attention is qwen35_linear_attn.install_device_aware_linear_attention is legacy_linear_attn.install_device_aware_linear_attention
    assert qwen35.fused_linear_attention_available is qwen35_linear_attn.fused_linear_attention_available is legacy_linear_attn.fused_linear_attention_available
    assert qwen35.install_expanded_gqa_attention is qwen35_gqa.install_expanded_gqa_attention is legacy_gqa.install_expanded_gqa_attention
    assert qwen35.fused_kernel_supports_gqa is qwen35_gqa.fused_kernel_supports_gqa is legacy_gqa.fused_kernel_supports_gqa

    # Model loader
    import distillkit.main as main
    assert models.load_student_model is main.load_student_model


def test_model_registry_resolution():
    # 1. Widened residual stream takes precedence
    widened_cfg = SimpleNamespace(residual_stream={"num_branches": 2}, sidecar=None, model_auto_class="AutoModelForCausalLM")
    assert models.resolve_student_class(widened_cfg) is qwen35.Qwen35WidenedForCausalLM

    # 2. Sidecar takes precedence when configured
    sidecar_cfg = SimpleNamespace(residual_stream=None, sidecar={"layer_index": 1}, model_auto_class="AutoModelForCausalLM")
    assert models.resolve_student_class(sidecar_cfg) is qwen35.Qwen35SidecarForCausalLM

    # 3. Standard Hugging Face fallback
    standard_cfg = SimpleNamespace(residual_stream=None, sidecar=None, model_auto_class="AutoModelForCausalLM")
    import transformers
    assert models.resolve_student_class(standard_cfg) is transformers.AutoModelForCausalLM

    # 4. Unknown model class raises ValueError
    bad_cfg = SimpleNamespace(residual_stream=None, sidecar=None, model_auto_class="NonExistentModelClassXYZ")
    with pytest.raises(ValueError, match="not found in transformers"):
        models.resolve_student_class(bad_cfg)


def test_custom_registry_extension():
    class DummyModel:
        pass

    registry.register_student_model("CustomRegisteredModel", DummyModel)
    cfg = SimpleNamespace(residual_stream=None, sidecar=None, model_auto_class="CustomRegisteredModel")
    assert models.resolve_student_class(cfg) is DummyModel
