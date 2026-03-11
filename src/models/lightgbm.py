"""
Functions for the training, evaluation, prediction, and storage for a LightGBM
binary classifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import re

import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


CatFeatureSpec = Sequence[Union[int, str]]


def _infer_cat_features(X: Any) -> Optional[list[str]]:
    if not isinstance(X, pd.DataFrame):
        return None

    cat_cols = X.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    return cat_cols if cat_cols else None


def _sanitize_lgbm_feature_names(columns):
    clean = []
    for c in columns:
        c2 = re.sub(r"[^0-9a-zA-Z_]+", "_", str(c))
        clean.append(c2 if c2 else "f")

    seen = {}
    final = []
    for c in clean:
        if c not in seen:
            seen[c] = 0
            final.append(c)
        else:
            seen[c] += 1
            final.append(f"{c}__{seen[c]}")

    return final


def _as_dataset(
    X: Any,
    y: Optional[Any] = None,
    cat_features: Optional[CatFeatureSpec] = None
) -> lgb.Dataset:

    if isinstance(X, pd.DataFrame):
        X = X.copy()
        X.columns = _sanitize_lgbm_feature_names(X.columns)

    if cat_features is None:
        cat_features = _infer_cat_features(X)

    if y is None:
        return lgb.Dataset(
            X,
            free_raw_data=False,
            categorical_feature=cat_features
        )

    return lgb.Dataset(
        X,
        label=y,
        free_raw_data=False,
        categorical_feature=cat_features
    )


def fit_lightgbm(
    train_X: Any,
    train_y: Any,
    base_params: Dict[str, object],
    cat_features: Optional[CatFeatureSpec] = None
) -> lgb.Booster:
    params: Dict[str, object] = dict(base_params)

    del params["one_hot"]
    num_boost_round = int(params["num_boost_round"])
    del params["num_boost_round"]

    train_ds = _as_dataset(train_X, train_y, cat_features=cat_features)

    model = lgb.train(
        params=params,
        train_set=train_ds,
        num_boost_round=num_boost_round,
        valid_sets=None,
        callbacks=None
    )
    return model


def evaluate_lightgbm(
    model: lgb.Booster,
    val_X: Any,
    val_y: Union[pd.Series, Any]
) -> pd.DataFrame:
    y_proba = model.predict(val_X)

    y_true = val_y.to_numpy() if isinstance(val_y, pd.Series) else val_y
    auc = roc_auc_score(y_true, y_proba)

    return pd.DataFrame([[auc]], columns=["AUC"])


def predict_lightgbm(
    model: lgb.Booster,
    test_X: Any
) -> pd.Series:
    proba = model.predict(test_X)
    return pd.Series(proba, name="proba")


def store_lightgbm(
    model: lgb.Booster,
    filepath: Path,
    model_name: str
) -> None:
    path = filepath / f"{model_name}.txt"
    model.save_model(str(path))
