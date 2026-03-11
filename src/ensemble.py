"""
Function for the evaluation for ensembles.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate_ensemble(y_true, y_proba) -> float:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    return float(roc_auc_score(y_true, y_proba))
