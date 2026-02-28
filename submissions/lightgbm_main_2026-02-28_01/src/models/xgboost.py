"""
Functions for the training, evaluation, prediction, and storage for an XGBoost
binary classifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

import xgboost as xgb
from xgboost import Booster
from sklearn.metrics import roc_auc_score


def _as_dmatrix(
    X: Any,
    y: Optional[Any] = None,
) -> xgb.DMatrix:
    if y is None:
        return xgb.DMatrix(X)
    return xgb.DMatrix(X, label=y)


def fit_xgboost(
    train_X: Any,
    train_y: Any,
    base_params: Dict[str, object],
) -> Booster:
    params = dict(base_params)

    num_boost_round = 200
    verbose_eval = False

    dtrain = _as_dmatrix(train_X, train_y)

    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        verbose_eval=verbose_eval,
    )

    return bst


def evaluate_xgboost(
    model: Booster,
    val_X: Any,
    val_y: Union[pd.Series, Any],
) -> pd.DataFrame:
    dval = _as_dmatrix(val_X)
    y_proba = model.predict(dval)

    y_true = val_y.to_numpy() if isinstance(val_y, pd.Series) else val_y
    auc = roc_auc_score(y_true, y_proba)

    return pd.DataFrame([[auc]], columns=["AUC"])


def predict_xgboost(
    model: Booster,
    test_X: Any,
) -> pd.Series:
    dtest = _as_dmatrix(test_X)
    proba = model.predict(dtest)
    return pd.Series(proba, name="proba")


def store_xgboost(
    model: Booster,
    filepath: Path,
    model_name: str,
) -> None:
    path = filepath / f"{model_name}.json"
    model.save_model(str(path))
