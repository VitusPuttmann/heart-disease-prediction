"""
Function for the evaluation for ensembles.
"""

from pathlib import Path

import pandas as pd

from sklearn.metrics import roc_auc_score


def evaluate_ensemble(
    y_true: pd.Series,
    y_proba: pd.Series
) -> float:

    return float(roc_auc_score(y_true, y_proba))
