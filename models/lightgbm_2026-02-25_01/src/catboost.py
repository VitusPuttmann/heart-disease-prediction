"""
Functions for the training, evaluation, prediction, and storage for a CatBoost
binary classifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import pandas as pd

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score


CatFeatureSpec = Sequence[Union[int, str]]


def _infer_cat_features(X: Any) -> Optional[list[str]]:
    if not isinstance(X, pd.DataFrame):
        return None

    cat_cols = X.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    return cat_cols if cat_cols else None


def _as_pool(
    X: Any,
    y: Optional[Any] = None,
    cat_features: Optional[CatFeatureSpec] = None
) -> Pool:
    if cat_features is None:
        cat_features = _infer_cat_features(X)

    if y is None:
        return Pool(X, cat_features=cat_features)
    return Pool(X, label=y, cat_features=cat_features)


def fit_catboost(
    train_X: Any,
    train_y: Any,
    base_params: Dict[str, object],
    cat_features: Optional[CatFeatureSpec] = None
) -> CatBoostClassifier:
    params: Dict[str, object] = dict(base_params)
    params.pop("one_hot", None)

    params.setdefault("iterations", 200)
    params.setdefault("verbose", False)

    train_pool = _as_pool(train_X, train_y, cat_features=cat_features)

    model = CatBoostClassifier(**params)
    model.fit(train_pool)

    return model


def evaluate_catboost(
    model: CatBoostClassifier,
    val_X: Any,
    val_y: Union[pd.Series, Any],
    cat_features: Optional[CatFeatureSpec] = None
) -> pd.DataFrame:
    val_pool = _as_pool(val_X, cat_features=cat_features)
    y_proba = model.predict_proba(val_pool)[:, 1]

    y_true = val_y.to_numpy() if isinstance(val_y, pd.Series) else val_y
    auc = roc_auc_score(y_true, y_proba)

    return pd.DataFrame([[auc]], columns=["AUC"])


def predict_catboost(
    model: CatBoostClassifier,
    test_X: Any,
    cat_features: Optional[CatFeatureSpec] = None
) -> pd.Series:
    test_pool = _as_pool(test_X, cat_features=cat_features)
    proba = model.predict_proba(test_pool)[:, 1]
    
    return pd.Series(proba, name="proba")


def store_catboost(
    model: CatBoostClassifier,
    filepath: Path,
    model_name: str,
) -> None:
    path = filepath / f"{model_name}.cbm"
    model.save_model(str(path))
