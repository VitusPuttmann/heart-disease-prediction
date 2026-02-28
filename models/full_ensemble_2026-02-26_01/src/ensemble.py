"""
Function for the implementation and evaluation for ensembles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score


def evaluate_ensemble(y_true, y_proba) -> float:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    return float(roc_auc_score(y_true, y_proba))


@dataclass
class StackingModel:
    kind: Literal["ridge", "logistic"]
    model: object
    clip_proba: float = 1e-6


def _clip(p: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def fit_stacking_regression(
    oof_pred: pd.DataFrame,
    y: pd.Series,
    kind: Literal["ridge", "logistic"] = "ridge",
    alpha: float = 1.0,
    clip_proba: float = 1e-6,
) -> StackingModel:
    """
    Meta-learner trained on OOF base-model probabilities.

    ridge:   Ridge regression on logit(p); final proba = sigmoid(pred)
    logistic: LogisticRegression directly on p
    """
    X = oof_pred.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=int)

    if kind == "ridge":
        Xc = _logit(_clip(X, clip_proba))
        m = Ridge(alpha=float(alpha), fit_intercept=True, random_state=0)
        m.fit(Xc, yv)
        return StackingModel(kind="ridge", model=m, clip_proba=clip_proba)

    if kind == "logistic":
        m = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
        )
        m.fit(X, yv)
        return StackingModel(kind="logistic", model=m, clip_proba=clip_proba)

    raise ValueError(f"Unknown meta-model kind: {kind}")


def predict_stacking_regression(
    stack: StackingModel,
    base_pred: pd.DataFrame,
) -> pd.Series:
    X = base_pred.to_numpy(dtype=float)

    if stack.kind == "ridge":
        Xc = _logit(_clip(X, stack.clip_proba))
        z = np.asarray(stack.model.predict(Xc), dtype=float)  # type: ignore
        p = 1.0 / (1.0 + np.exp(-z))
        return pd.Series(p, index=base_pred.index, name="stacked_proba")

    # logistic
    p = np.asarray(stack.model.predict_proba(X)[:, 1], dtype=float)  # type: ignore
    return pd.Series(p, index=base_pred.index, name="stacked_proba")
