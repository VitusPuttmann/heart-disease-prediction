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
    """
    LightGBM does not allow special JSON characters in feature names.
    Replace everything except [0-9a-zA-Z_] with underscore and deduplicate.
    """
    clean = []
    for c in columns:
        c2 = re.sub(r"[^0-9a-zA-Z_]+", "_", str(c))
        clean.append(c2 if c2 else "f")

    # ensure uniqueness
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
    cat_features: Optional[CatFeatureSpec] = None,
) -> lgb.Dataset:  # type: ignore

    # --- sanitize feature names if DataFrame ---
    if isinstance(X, pd.DataFrame):
        X = X.copy()
        X.columns = _sanitize_lgbm_feature_names(X.columns)

    if cat_features is None:
        cat_features = _infer_cat_features(X)

    if y is None:
        return lgb.Dataset(  # type: ignore
            X,
            free_raw_data=False,
            categorical_feature=cat_features,  # type: ignore
        )

    return lgb.Dataset(  # type: ignore
        X,
        label=y,
        free_raw_data=False,
        categorical_feature=cat_features,  # type: ignore
    )


def build_monotone_constraints(
    X: Any,
    mono_spec: Dict[str, int],
) -> Optional[list[int]]:
    if not isinstance(X, pd.DataFrame):
        return None
    
    feature_names = list(X.columns)
    cons: list[int] = []

    for col in feature_names:
        sign = 0
        for base, s in mono_spec.items():
            if col == base or col.startswith(base + "_"):
                sign = int(s)
                break
        cons.append(sign)

    return cons


def fit_lightgbm(
    train_X: Any,
    train_y: Any,
    base_params: Dict[str, object],
    cat_features: Optional[CatFeatureSpec] = None,
    mono_spec: Optional[Dict[str, int]] = None,
) -> lgb.Booster:  # type: ignore
    params: Dict[str, object] = dict(base_params)

    params.pop("one_hot", None)

    params.setdefault("objective", "binary")
    params.setdefault("metric", "auc")
    params.setdefault("verbosity", -1)

    # --- monotone constraints ---
    if mono_spec is not None and params.get("monotone_constraints") is None:
        cons = build_monotone_constraints(train_X, mono_spec)
        if cons is not None:
            params["monotone_constraints"] = cons
            params.setdefault("monotone_constraints_method", "advanced")
    # ----------------------------

    num_boost_round = int(params.pop("num_boost_round", params.pop("n_estimators", 200)))  # type: ignore
    early_stopping_rounds = params.pop("early_stopping_rounds", None)

    train_ds = _as_dataset(train_X, train_y, cat_features=cat_features)

    callbacks = []
    if early_stopping_rounds is not None:
        callbacks.append(lgb.early_stopping(int(early_stopping_rounds), verbose=False))  # type: ignore

    model = lgb.train(  # type: ignore
        params=params,
        train_set=train_ds,
        num_boost_round=num_boost_round,
        valid_sets=None,
        callbacks=callbacks if callbacks else None,
    )
    return model


def evaluate_lightgbm(
    model: lgb.Booster, # type: ignore
    val_X: Any,
    val_y: Union[pd.Series, Any],
    cat_features: Optional[CatFeatureSpec] = None,
) -> pd.DataFrame:
    y_proba = model.predict(val_X)

    y_true = val_y.to_numpy() if isinstance(val_y, pd.Series) else val_y
    auc = roc_auc_score(y_true, y_proba) # type: ignore

    return pd.DataFrame([[auc]], columns=["AUC"])


def predict_lightgbm(
    model: lgb.Booster, # type: ignore
    test_X: Any,
    cat_features: Optional[CatFeatureSpec] = None,
) -> pd.Series:
    proba = model.predict(test_X)
    return pd.Series(proba, name="proba")


def store_lightgbm(
    model: lgb.Booster, # type: ignore
    filepath: Path,
    model_name: str,
) -> None:
    path = filepath / f"{model_name}.txt"
    model.save_model(str(path))
