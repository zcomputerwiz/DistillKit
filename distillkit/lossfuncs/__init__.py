from distillkit.lossfuncs.common import LossFunctionBase
from distillkit.lossfuncs.cross_entropy import CrossEntropyLoss, AssistantCrossEntropyLoss
from distillkit.lossfuncs.hidden_state import HiddenStateCosineLoss, HiddenStateMSELoss
from distillkit.lossfuncs.hingeloss import HingeLoss, sparse_hinge_loss
from distillkit.lossfuncs.jsd import JSDLoss, sparse_js_div
from distillkit.lossfuncs.kl import KLDLoss, sparse_kl_div
from distillkit.lossfuncs.logistic_ranking import (
    LogisticRankingLoss,
    sparse_logistic_ranking_loss,
)
from distillkit.lossfuncs.tvd import TVDLoss, sparse_tvd
from distillkit.missing_probability import MissingProbabilityHandling

ALL_LOSS_CLASSES = [
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

from distillkit.lossfuncs.registry import (
    register_loss_function,
    get_loss_function_class,
    create_loss_function,
    list_registered_loss_functions,
)

__all__ = [
    "sparse_kl_div",
    "sparse_js_div",
    "sparse_tvd",
    "sparse_hinge_loss",
    "sparse_logistic_ranking_loss",
    "MissingProbabilityHandling",
    "KLDLoss",
    "JSDLoss",
    "TVDLoss",
    "HingeLoss",
    "LogisticRankingLoss",
    "HiddenStateCosineLoss",
    "HiddenStateMSELoss",
    "CrossEntropyLoss",
    "AssistantCrossEntropyLoss",
    "LossFunctionBase",
    "ALL_LOSS_CLASSES",
    "register_loss_function",
    "get_loss_function_class",
    "create_loss_function",
    "list_registered_loss_functions",
]
