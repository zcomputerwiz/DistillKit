"""Unit tests for the dynamic loss function registry and factory."""

import pytest
import torch

from distillkit.configuration import LossFunction, LossFunctionConfig
from distillkit.lossfuncs import (
    LossFunctionBase,
    create_loss_function,
    get_loss_function_class,
    list_registered_loss_functions,
    register_loss_function,
)
from distillkit.lossfuncs.kl import KLDLoss


class CustomTestLoss(LossFunctionBase):
    @classmethod
    def name(cls) -> str:
        return "custom_test_loss"

    def __init__(self, multiplier: float = 1.0, **kwargs):
        self.multiplier = multiplier

    def __call__(self, student_outputs, signal, mask=None, **kwargs):
        return torch.tensor(42.0) * self.multiplier


def test_builtin_losses_registered():
    registered = list_registered_loss_functions()
    assert "kl" in registered
    assert "jsd" in registered
    assert "cross_entropy" in registered
    assert "assistant_cross_entropy" in registered
    assert "hs_cosine" in registered
    assert "hs_mse" in registered
    assert "hinge" in registered
    assert "logistic_ranking" in registered
    assert "tvd" in registered


def test_get_loss_function_class():
    cls = get_loss_function_class("kl")
    assert cls is KLDLoss

    with pytest.raises(KeyError):
        get_loss_function_class("non_existent_loss")


def test_create_builtin_loss():
    cfg = LossFunctionConfig(
        function=LossFunction.KL,
        weight=1.0,
        temperature=2.0,
    )
    loss_fn = create_loss_function(cfg)
    assert isinstance(loss_fn, KLDLoss)
    assert loss_fn.temperature == 2.0


def test_register_and_create_custom_loss():
    register_loss_function(CustomTestLoss)
    assert "custom_test_loss" in list_registered_loss_functions()
    assert get_loss_function_class("custom_test_loss") is CustomTestLoss

    class CustomConfig:
        function = "custom_test_loss"
        weight = 0.5
        def model_dump(self, **kwargs):
            return {"multiplier": 2.5}

    custom_instance = create_loss_function(CustomConfig())
    assert isinstance(custom_instance, CustomTestLoss)
    assert custom_instance.multiplier == 2.5
