"""
Functions for the training, evaluation and prediction for an isotonic-calibrated
elastic net logistic regression model.
"""

import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


LRENI_PARAMS = {
    "solver": "saga",
    "max_iter": 500,
    "C": 1.0,
    "tol": 1e-3,
    "l1_ratio": 0.5,
    "class_weight": None,
    "random_state": 907364,
}


def fit_logistic_regression_en_iso(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    base_params: dict,
    cv: int = 5,
) -> CalibratedClassifierCV:
    """
    Fit an elastic net logistic regression model and calibrate probabilities
    using isotonic regression via cross-validation.
    """

    base_model = LogisticRegression(**base_params)

    model = CalibratedClassifierCV(
        estimator=base_model,
        method="isotonic",
        cv=cv,
    )

    model.fit(train_X, train_y)

    return model


def evaluate_logistic_regression_en_iso(
    model: CalibratedClassifierCV,
    val_X: pd.DataFrame,
    val_y: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate an isotonic-calibrated elastic net logistic regression model.
    """

    y_pred = model.predict(val_X)

    acc = accuracy_score(val_y, y_pred)

    return pd.DataFrame([[acc]], columns=["accuracy"])


def predict_logistic_regression_en_iso(
    model: CalibratedClassifierCV,
    test_X: pd.DataFrame,
) -> pd.Series:
    """
    Predict calibrated probabilities for a test dataset based on a
    isotonic-calibrated elastic net logistic regression model.
    """

    proba = model.predict_proba(test_X)[:, 1]
    
    return pd.Series(proba, name="proba")
