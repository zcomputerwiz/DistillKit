"""Regression gates: the student loader preserves a padded head that cached signals require.

A teacher cache stores top-k signals normalized over the teacher's full (padded)
head, e.g. 248,320 entries for a 248,077-entry tokenizer vocabulary. The loader
must not shrink such a head just because the tokenizer has fewer real entries,
and must reject heads that genuinely cannot cover the cached IDs.
"""

from types import SimpleNamespace

import pytest
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from distillkit.configuration import DistillationRunConfig
from distillkit.main import load_student_model


def _tiny_model_dir(tmp_path, vocab_size):
    config = Qwen3_5TextConfig(
        vocab_size=vocab_size, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_conv_kernel_dim=4, full_attention_interval=2,
        tie_word_embeddings=True, max_position_embeddings=64, pad_token_id=0,
        eos_token_id=3, use_cache=False,
    )
    path = tmp_path / f"model-{vocab_size}"
    Qwen3_5ForCausalLM(config).eval().save_pretrained(path)
    return str(path)


def _run_config(tmp_path, model_path, sidecar=None):
    payload = {
        "model": model_path,
        "dataset": {},
        # cache_path satisfies the config validator; load_student_model never reads it.
        "teacher": {"kind": "dataset", "cache_path": str(tmp_path / "cache")},
        "sequence_length": 8,
        "output_path": str(tmp_path / "out"),
        "use_flash_attention": False,
    }
    if sidecar is not None:
        payload["sidecar"] = sidecar
    return DistillationRunConfig.model_validate(payload)


def test_cached_signal_preserves_padded_head_without_sidecar(tmp_path):
    model_path = _tiny_model_dir(tmp_path, vocab_size=64)
    config = _run_config(tmp_path, model_path)
    model = load_student_model(config, tokenizer_vocab_size=60, signal_vocab_size=64)
    assert model.get_input_embeddings().weight.shape[0] == 64
    assert model.config.vocab_size == 64


def test_undersized_head_rejected_for_cached_signal(tmp_path):
    model_path = _tiny_model_dir(tmp_path, vocab_size=50)
    config = _run_config(tmp_path, model_path)
    with pytest.raises(ValueError, match="smaller than the cached"):
        load_student_model(config, tokenizer_vocab_size=48, signal_vocab_size=64)


def test_disabled_sidecar_control_preserves_padded_head(tmp_path):
    model_path = _tiny_model_dir(tmp_path, vocab_size=64)
    config = _run_config(tmp_path, model_path, sidecar={"enabled": False})
    model = load_student_model(config, tokenizer_vocab_size=60, signal_vocab_size=64)
    assert model.get_input_embeddings().weight.shape[0] == 64


def test_plain_tokenizer_resize_behavior_unchanged_without_signal(tmp_path):
    # No cached-signal requirement: the historical resize-to-tokenizer path stays.
    model_path = _tiny_model_dir(tmp_path, vocab_size=64)
    config = _run_config(tmp_path, model_path)
    model = load_student_model(config, tokenizer_vocab_size=50, signal_vocab_size=None)
    assert model.get_input_embeddings().weight.shape[0] == 50


def test_online_teacher_signal_reaches_resize_not_cache_error(tmp_path):
    # An online teacher sets signal_vocab_size == tokenizer_vocab_size. A student
    # head smaller than that must grow via the ordinary resize path, not trip the
    # cache-only "smaller than the cached signal vocabulary" error (which would
    # block previously supported vocabulary expansion).
    model_path = _tiny_model_dir(tmp_path, vocab_size=50)
    config = _run_config(tmp_path, model_path)
    model = load_student_model(config, tokenizer_vocab_size=60, signal_vocab_size=60)
    assert model.get_input_embeddings().weight.shape[0] == 60


def test_missing_flash_attn_fails_before_loading_with_actionable_message(monkeypatch):
    """use_flash_attention defaults true, but flash_attn has no Windows wheels.

    Without this guard the run dies inside from_pretrained after the dataset and
    teacher cache are already built, and the message does not mention that the flag
    is also what selects bfloat16.
    """
    import importlib.util

    from distillkit import main as main_module

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "flash_attn" else real_find_spec(name, *a, **k),
    )

    config = SimpleNamespace(
        functionary_packing=False,
        sidecar=None,
        model_auto_class="AutoModelForCausalLM",
        trust_remote_code=False,
        train_model="does-not-matter",
        use_flash_attention=True,
        model_kwargs={},
    )
    with pytest.raises(RuntimeError, match="flash_attn is not installed"):
        main_module.load_student_model(config, tokenizer_vocab_size=32, signal_vocab_size=32)


def test_no_stale_accelerator_reads_after_training_args_are_built():
    """Reading the pre-SFTConfig Accelerator later raises, and only at runtime.

    `trl.SFTConfig(...)` can call `AcceleratorState._reset_state()`, after which the
    `Accelerator` built earlier in `do_distill` raises AttributeError on any state
    access. That bit twice -- once at the optimizer-backend check and once at the
    resident-table guard -- each time only after the 4B student had already loaded.

    Rather than mock the whole run, assert the source does not reach for accelerator
    state after the training arguments exist: past that line, `training_arguments`
    carries the same values from a source that cannot go stale.
    """
    import inspect

    from distillkit import main as main_module

    source = inspect.getsource(main_module.do_distill)
    build_index = source.index("training_arguments = trl.SFTConfig(")
    after = source[build_index:]
    offenders = [
        line.strip()
        for line in after.splitlines()
        if "accelerator." in line and not line.strip().startswith("#")
    ]
    assert not offenders, (
        "accelerator state read after SFTConfig construction: " + "; ".join(offenders)
    )


def test_bf16_training_arg_loads_the_student_in_bfloat16(tmp_path):
    """Only the flash-attention branch used to set a dtype.

    With `use_flash_attention: false` (no Windows wheels) the dtype fell through to
    whatever the checkpoint config named, so a checkpoint that names none loaded a
    4.27B student in fp32 -- 17.2 GiB of weights instead of 8.5 -- while
    `training_args.bf16` sat right there saying otherwise.
    """
    model_path = _tiny_model_dir(tmp_path, vocab_size=64)
    config = _run_config(tmp_path, model_path)
    config.training_args = {"bf16": True}
    model = load_student_model(config, tokenizer_vocab_size=60, signal_vocab_size=64)
    assert model.get_input_embeddings().weight.dtype is __import__("torch").bfloat16


def test_no_precision_flag_keeps_the_default_dtype(tmp_path):
    model_path = _tiny_model_dir(tmp_path, vocab_size=64)
    config = _run_config(tmp_path, model_path)
    model = load_student_model(config, tokenizer_vocab_size=60, signal_vocab_size=64)
    assert model.get_input_embeddings().weight.dtype is __import__("torch").float32
