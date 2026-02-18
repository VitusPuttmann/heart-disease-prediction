"""
Functions for the training, evaluation, prediction, and storage for a logistic
regression model.
"""

from pathlib import Path

import pandas as pd

import statsmodels.api as sm

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score


def store_regression_table(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    output_dir: Path
):
    X = train_X.copy()

    cat_cols = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    bool_cols = X.select_dtypes(
        include=["bool"]
    ).columns.tolist()
    for c in bool_cols:
        X[c] = X[c].astype(int)
    
    X = sm.add_constant(X)
    
    model = sm.GLM(train_y, X, family=sm.families.Binomial())
    res = model.fit()

    coef_table = res.summary2().tables[1]
    
    coef_table.to_csv(output_dir / "reg_table.csv")


def fit_logistic_regression(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    base_params: dict,
    cv: int = 5,
) -> CalibratedClassifierCV:
    """
    Fit a logistic regression model and calibrate probabilities using isotonic
    regression via cross-validation.
    """

    base_model = LogisticRegression(**base_params)

    model = CalibratedClassifierCV(
        estimator=base_model,
        method="isotonic",
        cv=cv,
    )

    model.fit(train_X, train_y)

    return model


def evaluate_logistic_regression(
    model: CalibratedClassifierCV,
    val_X: pd.DataFrame,
    val_y: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate a logistic regression model.
    """

    y_pred_proba = model.predict_proba(val_X)[:, 1]
    auc = roc_auc_score(val_y, y_pred_proba)

    return pd.DataFrame([[auc]], columns=["AUC"])


def predict_logistic_regression(
    model: CalibratedClassifierCV,
    test_X: pd.DataFrame,
) -> pd.Series:
    """
    Predict calibrated probabilities for a test dataset based on a logistic
    regression model.
    """

    proba = model.predict_proba(test_X)[:, 1]
    
    return pd.Series(proba, name="proba")


def store_logistic_regression(
    model: CalibratedClassifierCV,
    filepath: Path,
    model_name: str
) -> None:
    """
    Store a logistic regression model.
    """

    path = filepath / f"{model_name}.joblib"

    joblib.dump(model, path)
