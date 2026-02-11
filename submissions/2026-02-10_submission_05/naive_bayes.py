"""
Functions for the training, evaluation, and prediction for a Naive Bayes
clasifier.
"""

import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score


def fit_naive_bayes(
        train_X: pd.DataFrame,
        train_y: pd.Series,
        params: None = None
    ) -> GaussianNB:
    """
    Fit a Gaussian Naive Bayse classifier.
    """

    model = GaussianNB()
    model.fit(train_X, train_y)

    return model


def evaluate_naive_bayes(
        model: GaussianNB,
        val_X: pd.DataFrame,
        val_y: pd.Series
    ) -> pd.DataFrame:
    """
    Evaluate a previously trained Naive Bayes model.
    """

    y_pred = model.predict(val_X)
    y_prob = model.predict_proba(val_X)[:, 1]

    accuracy = accuracy_score(val_y, y_pred)
    auc = roc_auc_score(val_y, y_prob)

    scores = pd.DataFrame(
        [[accuracy, auc]],
        columns=["accuracy", "auc"]
    )

    return scores


def predict_naive_bayes(
        model: GaussianNB,
        test_X: pd.DataFrame
    ) -> pd.Series:
    """
    Predict values for a test dataset based on a previously trained Naive
    Bayes model.
    """

    y_pred = model.predict_proba(test_X)[:, 1]

    return pd.Series(y_pred)
