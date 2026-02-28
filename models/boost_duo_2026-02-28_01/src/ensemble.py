# src/ensemble.py
"""
Function for the implementation and evaluation for ensembles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def evaluate_ensemble(y_true, y_proba) -> float:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    return float(roc_auc_score(y_true, y_proba))


@dataclass
class StackingModel:
    kind: Literal["ridge", "logistic"]
    model: object
    clip_proba: float = 1e-6
    feature_names: tuple[str, ...] = ()          # enforce column order at predict time
    use_logit_features: bool = True              # whether meta-features are logit(p)


def _clip(p: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def _meta_features(X: np.ndarray, clip_proba: float, use_logit: bool) -> np.ndarray:
    if not use_logit:
        return X
    return _logit(_clip(X, clip_proba))


def fit_stacking_regression(
    oof_pred: pd.DataFrame,
    y: pd.Series,
    kind: Literal["ridge", "logistic"] = "ridge",
    alpha: float = 1.0,
    clip_proba: float = 1e-6,
    standardize: bool = True,          # NEW: scale meta-features
    use_logit_features: bool = True,   # NEW: also use logits for logistic meta-model
    logistic_C: float | None = None,   # NEW: allow explicit C; else derive from alpha
) -> StackingModel:
    """
    Meta-learner trained on OOF base-model probabilities.

    - Features: logit(p) by default (recommended).
    - ridge: Ridge regression on meta-features; final proba = sigmoid(pred)
    - logistic: LogisticRegression on meta-features; final proba = predict_proba
    """
    X = oof_pred.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=int)
    feat_names = tuple(oof_pred.columns.tolist())

    Xc = _meta_features(X, clip_proba=clip_proba, use_logit=use_logit_features)

    if kind == "ridge":
        # Ridge is sensitive to scale -> standardize by default
        steps = []
        if standardize:
            steps.append(("scaler", StandardScaler()))
        steps.append(("ridge", Ridge(alpha=float(alpha), fit_intercept=True, random_state=0)))
        m: object = Pipeline(steps)
        m.fit(Xc, yv)
        return StackingModel(
            kind="ridge",
            model=m,
            clip_proba=clip_proba,
            feature_names=feat_names,
            use_logit_features=use_logit_features,
        )

    if kind == "logistic":
        # If user provides alpha, interpret as inverse strength: C = 1/alpha (common convention)
        if logistic_C is None:
            logistic_C = 1.0 / max(float(alpha), 1e-12)
        steps = []
        if standardize:
            steps.append(("scaler", StandardScaler()))
        steps.append(
            ("logreg", LogisticRegression(
                penalty="l2",
                C=float(logistic_C),
                solver="lbfgs",
                max_iter=5000,
            ))
        )
        m = Pipeline(steps)
        m.fit(Xc, yv)
        return StackingModel(
            kind="logistic",
            model=m,
            clip_proba=clip_proba,
            feature_names=feat_names,
            use_logit_features=use_logit_features,
        )

    raise ValueError(f"Unknown meta-model kind: {kind}")


def predict_stacking_regression(stack: StackingModel, base_pred: pd.DataFrame) -> pd.Series:
    # Enforce training-time column order
    if stack.feature_names:
        missing = [c for c in stack.feature_names if c not in base_pred.columns]
        if missing:
            raise ValueError(f"Missing meta-features at predict time: {missing}")
        base_pred = base_pred.loc[:, list(stack.feature_names)]

    X = base_pred.to_numpy(dtype=float)
    Xc = _meta_features(X, clip_proba=stack.clip_proba, use_logit=stack.use_logit_features)

    if stack.kind == "ridge":
        z = np.asarray(stack.model.predict(Xc), dtype=float)  # type: ignore
        p = 1.0 / (1.0 + np.exp(-z))
        return pd.Series(p, index=base_pred.index, name="stacked_proba")

    # logistic
    p = np.asarray(stack.model.predict_proba(Xc)[:, 1], dtype=float)  # type: ignore
    return pd.Series(p, index=base_pred.index, name="stacked_proba")
