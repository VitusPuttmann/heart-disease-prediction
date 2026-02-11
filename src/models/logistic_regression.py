"""
Functions for the training, evaluation and prediction for a logistic regression
model.
"""

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


LR_PARAMS = {
    "solver" : "saga",
    "max_iter": 200,
    "C": 1.0,
    "tol": 1e-3,
    "l1_ratio": 0,
    "class_weight": None,
    "random_state": 907364
}


def fit_logistic_regression(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    params: dict,
) -> LogisticRegression:
    """
    Fit a logistic regression model.
    """

    model = LogisticRegression(**params)

    model.fit(train_X, train_y)

    return model


def evaluate_logistic_regression(
    model: LogisticRegression,
    val_X: pd.DataFrame,
    val_y: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate a previously trained logistic regression model.
    """

    y_pred = model.predict(val_X)
    y_proba = model.predict_proba(val_X)[:, 1]

    acc = accuracy_score(val_y, y_pred)

    return pd.DataFrame(
        [[acc]],
        columns=["accuracy"],
    )


def predict_logistic_regression(
    model: LogisticRegression,
    test_X: pd.DataFrame,
) -> pd.Series:
    """
    Predict values for a test dataset based on a previously trained logistic
    regression model.
    """

    proba = model.predict_proba(test_X)[:, 1]
    
    return pd.Series(proba)
