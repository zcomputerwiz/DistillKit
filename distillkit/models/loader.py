# Copyright 2025 Arcee AI & DistillKit Contributors
"""Model lifecycle, preparation, and configuration protocol for student models."""

import importlib.util
import logging
import re
from typing import Any

import torch
import transformers

from distillkit.configuration import DistillationRunConfig
from distillkit.monkey_patch_packing import monkey_patch_packing_for_model
from distillkit.models.registry import resolve_student_class

LOG = logging.getLogger(__name__)


def prepare_student_config(
    config: DistillationRunConfig, extra_kwargs: dict[str, Any]
) -> tuple[Any, str | None]:
    """Inspect and configure model text_config with sidecar/residual_stream settings.

    Returns the prepared text_config and any saved_sidecar_variant from checkpoint.
    """
    residual_stream = getattr(config, "residual_stream", None)
    config_kwargs = {
        key: extra_kwargs[key]
        for key in (
            "revision",
            "cache_dir",
            "local_files_only",
            "token",
            "subfolder",
            "trust_remote_code",
        )
        if key in extra_kwargs
    }
    stock_config = transformers.AutoConfig.from_pretrained(config.train_model, **config_kwargs)
    text_config = getattr(stock_config, "text_config", stock_config)
    saved_sidecar_variant = getattr(text_config, "sidecar_variant", None)

    if getattr(text_config, "residual_stream_enabled", False):
        if residual_stream is None:
            raise ValueError("Widened checkpoint requires its matching residual_stream run configuration")
        expected = (
            text_config.residual_stream_num_branches,
            text_config.residual_stream_lowrank,
            getattr(text_config, "residual_stream_sidecar", False),
            getattr(text_config, "residual_stream_routing", "widened"),
        )
        requested = (
            residual_stream.num_branches,
            residual_stream.lowrank,
            config.sidecar is not None,
            residual_stream.routing,
        )
        if expected != requested:
            raise ValueError(f"Widened checkpoint architecture {expected} differs from requested {requested}")
        if config.sidecar is not None and (
            text_config.sidecar_layer_index != config.sidecar.layer_index
            or text_config.sidecar_num_branches != config.sidecar.num_branches
            or text_config.sidecar_variant != config.sidecar.variant
            or getattr(text_config, "sidecar_table_mode", "donor") != config.sidecar.table_mode
            or (
                config.sidecar.variant == "ple_gated"
                and getattr(text_config, "sidecar_gate_directions", None) != config.sidecar.gate_directions
            )
        ):
            raise ValueError("Widened checkpoint sidecar architecture differs from requested sidecar")

    if config.sidecar is not None or residual_stream is not None:
        if config.sidecar is not None:
            if saved_sidecar_variant == "donor_reader":
                expected = (
                    text_config.sidecar_value_source,
                    text_config.sidecar_conv_source,
                    text_config.sidecar_reader_collapse,
                    text_config.sidecar_reader_single_stream,
                    getattr(text_config, "sidecar_reader_collapse_weights", None),
                )
                requested = (
                    config.sidecar.reader_value_source,
                    config.sidecar.reader_conv_source,
                    config.sidecar.reader_collapse,
                    config.sidecar.reader_single_stream,
                    config.sidecar.reader_collapse_weights,
                )
                if expected != requested:
                    raise ValueError(
                        f"Saved donor-reader arm {expected} differs from requested {requested}"
                    )
            text_config.sidecar_layer_index = config.sidecar.layer_index
            text_config.sidecar_num_branches = config.sidecar.num_branches
            text_config.sidecar_variant = config.sidecar.variant
            text_config.sidecar_gate_directions = config.sidecar.gate_directions
            text_config.sidecar_value_source = config.sidecar.reader_value_source
            text_config.sidecar_conv_source = config.sidecar.reader_conv_source
            text_config.sidecar_reader_collapse = config.sidecar.reader_collapse
            text_config.sidecar_reader_single_stream = config.sidecar.reader_single_stream
            text_config.sidecar_reader_collapse_weights = config.sidecar.reader_collapse_weights
            text_config.sidecar_reader_rho = config.sidecar.reader_rho
            text_config.sidecar_table_mode = config.sidecar.table_mode
            text_config.sidecar_ngram_vocab_size_base = config.sidecar.ngram_vocab_size_base
            text_config.sidecar_ple_embed_dim = config.sidecar.ple_embed_dim
        if residual_stream is not None:
            text_config.residual_stream_enabled = True
            text_config.residual_stream_num_branches = residual_stream.num_branches
            text_config.residual_stream_lowrank = residual_stream.lowrank
            text_config.residual_stream_sidecar = config.sidecar is not None
            text_config.residual_stream_routing = residual_stream.routing
            text_config.residual_stream_blend = residual_stream.blend
            text_config.residual_stream_learnable_blend = residual_stream.learnable_blend
        extra_kwargs["config"] = text_config

    return text_config, saved_sidecar_variant


def post_init_student_model(
    model: transformers.PreTrainedModel,
    config: DistillationRunConfig,
    saved_sidecar_variant: str | None,
) -> None:
    """Run architecture-specific parameter transplantation and initialization."""
    residual_stream = getattr(config, "residual_stream", None)
    if (
        config.sidecar is not None
        and config.sidecar.variant == "donor_reader"
        and saved_sidecar_variant != "donor_reader"
    ):
        from distillkit.donor_reader import initialise_transplant_reader

        LOG.info(
            "Initialized donor-reader arm: %s",
            initialise_transplant_reader(
                model,
                c1_reference=config.sidecar.reader_c1_reference,
                donor_reference=config.sidecar.reader_donor_reference,
            ),
        )

    if config.sidecar is not None and config.sidecar.variant == "donor_reader":
        reader = model.model.layers[config.sidecar.layer_index].sidecar.reader
        reader.enforce_trainability()
        reader_trainable = {
            name: parameter.numel()
            for name, parameter in reader.named_parameters()
            if parameter.requires_grad
        }
        expected_reader_trainable = (
            {"mixer.weight": 4 * reader.hidden_size, "rho.weight": 1}
            if reader.collapse == "mixer"
            else {"rho.weight": 1}
        )
        if reader_trainable != expected_reader_trainable:
            raise RuntimeError(
                "Donor-reader freeze contract violated: "
                f"expected {expected_reader_trainable}, got {reader_trainable}"
            )
        LOG.info("Donor-reader trainable parameters: %s", reader_trainable)

    if residual_stream is not None and residual_stream.init_from:
        from distillkit.borrowed_routing import initialise_widened_residual

        LOG.info(
            "Borrowed routing: %s",
            initialise_widened_residual(
                model, residual_stream.init_from, residual_stream.init_layer_map
            ),
        )


def align_student_embeddings(
    model: transformers.PreTrainedModel,
    tokenizer_vocab_size: int,
    signal_vocab_size: int | None,
    config: DistillationRunConfig,
) -> None:
    """Verify and adjust student vocabulary size against tokenizer and signal requirements."""
    residual_stream = getattr(config, "residual_stream", None)
    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    required_vocab_size = max(tokenizer_vocab_size, signal_vocab_size or 0)
    if config.sidecar is not None or residual_stream is not None:
        if model_vocab_size < required_vocab_size:
            raise ValueError(
                "Student head does not cover the tokenizer/signal vocabulary"
            )
    else:
        if (
            signal_vocab_size is not None
            and signal_vocab_size > tokenizer_vocab_size
            and model_vocab_size < signal_vocab_size
        ):
            raise ValueError(
                f"Student head ({model_vocab_size}) is smaller than the cached "
                f"signal vocabulary ({signal_vocab_size}); re-capture or use a "
                f"student whose head covers the cache vocabulary"
            )
        preserves_padded_head = (
            signal_vocab_size is not None
            and signal_vocab_size > tokenizer_vocab_size
            and model_vocab_size >= signal_vocab_size
        )
        if preserves_padded_head:
            LOG.info(
                f"Preserving padded student head of {model_vocab_size} entries "
                f"(tokenizer vocab {tokenizer_vocab_size}) to cover the cached "
                f"signal vocabulary"
            )
        elif (
            model_vocab_size != tokenizer_vocab_size
            or config.resize_embeddings_to_multiple_of
        ):
            model.resize_token_embeddings(
                tokenizer_vocab_size,
                pad_to_multiple_of=config.resize_embeddings_to_multiple_of,
            )
            new_model_vocab_size = model.get_input_embeddings().weight.shape[0]
            if new_model_vocab_size != model_vocab_size:
                LOG.info(
                    f"Resized model vocab size from {model_vocab_size} to {new_model_vocab_size}"
                )


def apply_freeze_rules(model: transformers.PreTrainedModel, config: DistillationRunConfig) -> None:
    """Freeze modules by exact name or regex pattern based on configuration."""
    if config.frozen_modules:
        module_set = set(config.frozen_modules)
        seen = set()
        for name, module in model.named_modules():
            if name in module_set:
                module.requires_grad_(False)
                seen.add(name)
        unseen = module_set - seen
        LOG.info(f"Froze {len(seen)} modules")
        if unseen:
            raise ValueError(f"Frozen modules not found in model: {', '.join(unseen)}")
    if config.frozen_res:
        num_frozen = 0
        frozen_res = [re.compile(s) for s in config.frozen_res]
        for name, param in model.named_parameters():
            if any(fre.search(name) for fre in frozen_res):
                param.requires_grad = False
                num_frozen += 1
        if num_frozen:
            print(f"Froze {num_frozen} tensors by regular expression")


def is_flash_attn_available() -> bool:
    """True if flash_attn is installed and its compiled C/CUDA extension loads."""
    if importlib.util.find_spec("flash_attn") is None:
        return False
    try:
        import flash_attn  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


def load_student_model(
    config: DistillationRunConfig,
    tokenizer_vocab_size: int,
    signal_vocab_size: int | None = None,
) -> transformers.PreTrainedModel:
    """Canonical loader and preparer for student models."""
    if config.functionary_packing:
        monkey_patch_packing_for_model(config.train_model)

    auto_cls = resolve_student_class(config)
    LOG.info(f"Loading model {config.train_model} with class {auto_cls}")

    extra_kwargs = {"trust_remote_code": config.trust_remote_code}
    if config.use_flash_attention:
        if importlib.util.find_spec("flash_attn") is None or not is_flash_attn_available():
            raise RuntimeError(
                "use_flash_attention is true but flash_attn is not installed "
                "or failed to load. Install a compatible wheel, or set "
                "use_flash_attention: false and set training_args.bf16 so the "
                "student still loads in bfloat16."
            )
        extra_kwargs["attn_implementation"] = "flash_attention_2"
        extra_kwargs["torch_dtype"] = torch.bfloat16
    extra_kwargs.update(config.model_kwargs)
    if "torch_dtype" not in extra_kwargs:
        if config.training_args.get("bf16"):
            extra_kwargs["torch_dtype"] = torch.bfloat16
        elif config.training_args.get("fp16"):
            extra_kwargs["torch_dtype"] = torch.float16

    _, saved_sidecar_variant = prepare_student_config(config, extra_kwargs)

    model = auto_cls.from_pretrained(
        config.train_model,
        **extra_kwargs,
    )
    LOG.info("Loaded model.")

    post_init_student_model(model, config, saved_sidecar_variant)
    align_student_embeddings(model, tokenizer_vocab_size, signal_vocab_size, config)
    apply_freeze_rules(model, config)

    return model


__all__ = [
    "is_flash_attn_available",
    "load_student_model",
    "prepare_student_config",
    "post_init_student_model",
    "align_student_embeddings",
    "apply_freeze_rules",
]
