# Copyright 2025 Arcee AI & DistillKit Contributors
"""Model architecture registry and resolver for student model classes.

Decouples custom model architectures (such as sidecar and widened architectures)
from main training orchestration logic, while maintaining full fallback to
standard Hugging Face transformers AutoModel classes.
"""

from __future__ import annotations

from typing import Any, Callable, Type
import logging
import torch.nn as nn
import transformers

LOG = logging.getLogger(__name__)

# Registry entries: list of (predicate_callable, model_class_or_factory)
_STUDENT_REGISTRY: list[tuple[Callable[[Any], bool], Type[nn.Module]]] = []
# Named registry: mapping of str -> model_class
_NAMED_REGISTRY: dict[str, Type[nn.Module]] = {}


def register_student_model(
    identifier: str | Callable[[Any], bool],
    model_class: Type[nn.Module],
) -> None:
    """Register a custom student model class either by name or by configuration predicate."""
    if callable(identifier) and not isinstance(identifier, str):
        _STUDENT_REGISTRY.append((identifier, model_class))
    else:
        _NAMED_REGISTRY[str(identifier)] = model_class


def _init_default_registry() -> None:
    """Lazy initialize built-in student models."""
    if _STUDENT_REGISTRY or _NAMED_REGISTRY:
        return

    from distillkit.models.qwen35.sidecar import Qwen35SidecarForCausalLM
    from distillkit.models.qwen35.widened import Qwen35WidenedForCausalLM

    register_student_model("Qwen35WidenedForCausalLM", Qwen35WidenedForCausalLM)
    register_student_model("Qwen35SidecarForCausalLM", Qwen35SidecarForCausalLM)

    # Predicate matching based on config flags:
    # 1. Widened residual stream takes priority when configured
    register_student_model(
        lambda cfg: getattr(cfg, "residual_stream", None) is not None,
        Qwen35WidenedForCausalLM,
    )
    # 2. Sidecar takes priority when configured
    register_student_model(
        lambda cfg: getattr(cfg, "sidecar", None) is not None,
        Qwen35SidecarForCausalLM,
    )


def resolve_student_class(config: Any) -> Type[nn.Module]:
    """Resolve the appropriate model class for a given student configuration.

    Evaluates registered custom model predicates, then named registry entries,
    and finally falls back to Hugging Face `transformers`.
    """
    _init_default_registry()

    # 1. Check predicates
    for predicate, model_cls in _STUDENT_REGISTRY:
        try:
            if predicate(config):
                return model_cls
        except Exception as exc:
            LOG.debug("Predicate check %s raised %s; skipping", predicate, exc)

    # 2. Check named registry
    model_auto_class = getattr(config, "model_auto_class", None)
    if model_auto_class and model_auto_class in _NAMED_REGISTRY:
        return _NAMED_REGISTRY[model_auto_class]

    # 3. Fallback to Hugging Face transformers
    if model_auto_class:
        auto_cls = getattr(transformers, model_auto_class, None)
        if auto_cls is not None:
            return auto_cls

    raise ValueError(
        f"Model class {model_auto_class} not found in transformers or model registry."
    )
