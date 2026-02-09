"""
Functions for the training, evaluation and prediction for a regression model.
"""

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def fit_regression(
        train_X: pd.DataFrame,
        train_y: pd.Series,
        params: None
    ) -> LinearRegression:
    """
    Fit a linear regression model.
    """

    model = LinearRegression()
    model.fit(train_X, train_y)

    return model


def evaluate_regression(
        model: LinearRegression,
        val_X: pd.DataFrame,
        val_y: pd.Series
    ) -> pd.DataFrame:
    """
    Evaluate a previously trained regression model and return MSE.
    """

    y_pred = model.predict(val_X)
    mse = mean_squared_error(val_y, y_pred)

    scores = pd.DataFrame(
        [[mse]],
        columns=["mse"]
    )

    return scores


def predict_regression(
        model: LinearRegression,
        test_X: pd.DataFrame,
    ) -> pd.Series:

    """
    Predict values for a test dataset based on a previously trained
    regression model.
    """

    y_pred = model.predict(test_X)

    return pd.Series(y_pred)
