# Copyright 2025 Arcee AI & DistillKit Contributors
"""Loss function registry and factory for DistillKit.

Enables dynamic registration and creation of loss functions without requiring
modifications to core enumerations or static class lists.
"""

from __future__ import annotations

from typing import Any, Callable, Type
import logging

from distillkit.configuration import LossFunctionConfig
from distillkit.lossfuncs.common import LossFunctionBase

LOG = logging.getLogger(__name__)

_LOSS_REGISTRY: dict[str, Type[LossFunctionBase]] = {}


def register_loss_function(
    identifier: str | Type[LossFunctionBase] | None = None,
    loss_cls: Type[LossFunctionBase] | None = None,
) -> Any:
    """Register a loss function class in the registry.

    Can be used as a decorator or a direct function call:
        @register_loss_function
        class MyLoss(LossFunctionBase): ...

        @register_loss_function("custom_loss")
        class MyLoss(LossFunctionBase): ...

        register_loss_function("custom_loss", MyLoss)
    """
    def _decorator(cls: Type[LossFunctionBase]) -> Type[LossFunctionBase]:
        name = identifier if isinstance(identifier, str) else cls.name()
        _LOSS_REGISTRY[name] = cls
        return cls

    if isinstance(identifier, type) and issubclass(identifier, LossFunctionBase):
        # Used as @register_loss_function without arguments
        cls = identifier
        _LOSS_REGISTRY[cls.name()] = cls
        return cls
    elif isinstance(identifier, str) and loss_cls is not None:
        # Used as register_loss_function("name", cls)
        _LOSS_REGISTRY[identifier] = loss_cls
        return loss_cls
    elif identifier is not None and not isinstance(identifier, str):
        raise TypeError(f"Invalid identifier type: {type(identifier)}")

    return _decorator


def _init_default_loss_registry() -> None:
    """Lazy initialize built-in loss functions in the registry."""
    if _LOSS_REGISTRY:
        return

    from distillkit.lossfuncs.cross_entropy import CrossEntropyLoss, AssistantCrossEntropyLoss
    from distillkit.lossfuncs.hidden_state import HiddenStateCosineLoss, HiddenStateMSELoss
    from distillkit.lossfuncs.hingeloss import HingeLoss
    from distillkit.lossfuncs.jsd import JSDLoss
    from distillkit.lossfuncs.kl import KLDLoss
    from distillkit.lossfuncs.logistic_ranking import LogisticRankingLoss
    from distillkit.lossfuncs.tvd import TVDLoss

    builtins = [
        KLDLoss,
        JSDLoss,
        TVDLoss,
        HingeLoss,
        LogisticRankingLoss,
        HiddenStateCosineLoss,
        HiddenStateMSELoss,
        CrossEntropyLoss,
        AssistantCrossEntropyLoss,
    ]
    for cls in builtins:
        register_loss_function(cls)


def get_loss_function_class(name: str) -> Type[LossFunctionBase]:
    """Retrieve a loss function class by name."""
    _init_default_loss_registry()
    if name not in _LOSS_REGISTRY:
        raise KeyError(f"Unknown loss function '{name}'. Registered: {list(_LOSS_REGISTRY.keys())}")
    return _LOSS_REGISTRY[name]


def create_loss_function(cfg: LossFunctionConfig) -> LossFunctionBase:
    """Factory to instantiate a loss function from configuration."""
    _init_default_loss_registry()
    name = cfg.function.value if hasattr(cfg.function, "value") else str(cfg.function)
    cls = get_loss_function_class(name)
    kwargs = cfg.model_dump(exclude=["function", "weight"], exclude_none=True)
    return cls(**kwargs)


def list_registered_loss_functions() -> list[str]:
    """Return a list of all registered loss function names."""
    _init_default_loss_registry()
    return sorted(_LOSS_REGISTRY.keys())


__all__ = [
    "register_loss_function",
    "get_loss_function_class",
    "create_loss_function",
    "list_registered_loss_functions",
]
