"""
Function for defining a registry for ML models.
"""

from typing import Any

from .spec import ModelSpec
from .catboost import (
    fit_catboost,
    evaluate_catboost,
    predict_catboost,
    store_catboost,
)
from .lightgbm import (
    fit_lightgbm,
    evaluate_lightgbm,
    predict_lightgbm,
    store_lightgbm,
)
from .logistic_regression import (
    fit_logistic_regression,
    evaluate_logistic_regression,
    predict_logistic_regression,
    store_logistic_regression,
)
from .neural_net import (
    fit_neuralnet,
    evaluate_neuralnet,
    predict_neuralnet,
    store_neural_net,
)
from .real_mlp import (
    fit_realmlp,
    evaluate_realmlp,
    predict_realmlp,
    store_realmlp,
)
from .xgboost import (
    fit_xgboost,
    evaluate_xgboost,
    predict_xgboost,
    store_xgboost,
)


def build_model_registry(
    LG_PARAMS: dict[str, Any],
    CB_PARAMS: dict[str, Any],
    LR_PARAMS: dict[str, Any],
    NN_PARAMS: dict[str, Any],
    RM_PARAMS: dict[str, Any],
    XG_PARAMS: dict[str, Any],
) -> dict[str, ModelSpec]:
    def _lgbm(params: dict[str, Any]) -> ModelSpec:
        return ModelSpec(
            params,
            fit_lightgbm,
            evaluate_lightgbm,
            predict_lightgbm,
            store_lightgbm,
        )

    return {
        "lightgbm_main": _lgbm(LG_PARAMS["main"]),
        "lightgbm_deep_expressive": _lgbm(LG_PARAMS["deep_expressive"]),
        "lightgbm_strong_reg": _lgbm(LG_PARAMS["strong_reg"]),
        "lightgbm_boost": _lgbm(LG_PARAMS["boost"]),
        "lightgbm_more_subs": _lgbm(LG_PARAMS["more_subs"]),
        "catboost": ModelSpec(
            CB_PARAMS,
            fit_catboost,
            evaluate_catboost,
            predict_catboost,
            store_catboost,
        ),
        "logistic_regression": ModelSpec(
            LR_PARAMS,
            fit_logistic_regression,
            evaluate_logistic_regression,
            predict_logistic_regression,
            store_logistic_regression,
        ),
        "neural_net": ModelSpec(
            NN_PARAMS,
            fit_neuralnet,
            evaluate_neuralnet,
            predict_neuralnet,
            store_neural_net,
        ),
        "real_mlp": ModelSpec(
            RM_PARAMS,
            fit_realmlp,
            evaluate_realmlp,
            predict_realmlp,
            store_realmlp,
        ),
        "xgboost": ModelSpec(
            XG_PARAMS,
            fit_xgboost,
            evaluate_xgboost,
            predict_xgboost,
            store_xgboost,
        ),
    }
