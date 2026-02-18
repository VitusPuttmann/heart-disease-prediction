"""
Pipeline for engineering features.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class SelectedInteractions(BaseEstimator, TransformerMixin):
    def __init__(self, interactions: List[Dict[str, Any]]):
        self.interactions = interactions
    
    @staticmethod
    def _cols_for_var(X: pd.DataFrame, var: str) -> List[str]:
        cols = []
        if var in X.columns:
            cols.append(var)
        
        prefix = f"{var}_"
        cols.extend([c for c in X.columns if c.startswith(prefix)])
        
        return cols
    
    @staticmethod
    def _origin_var(col: str) -> str:
        return col.split("_", 1)[0]

    def fit(self, X, y=None):
        self._added_cols_: List[str] = []
        self._plan_: List[tuple[str, str, str]] = []

        for spec in self.interactions:
            if spec["status"] != "include":
                continue

            v1, v2 = spec["vars"]
            base_name = spec["name"]

            cols1 = self._cols_for_var(X, v1)
            cols2 = self._cols_for_var(X, v2)

            for c1 in cols1:
                for c2 in cols2:
                    if self._origin_var(c1) == self._origin_var(c2):
                        continue

                    new_col = f"{base_name}__{c1}__x__{c2}"
                    self._plan_.append((new_col, c1, c2))
                    self._added_cols_.append(new_col)
        
        return self
    
    def transform(self, X):
        X_intacts = X.copy()

        for new_col, c1, c2 in self._plan_:
            X_intacts[new_col] = X_intacts[c1] * X_intacts[c2]
        
        return X_intacts

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = []
        
        return list(input_features) + getattr(self, "_added_cols_", [])


def build_preprocess_pipeline(
    num_cols: List[str],
    cat_cols: List[str],
    standardize: bool,
    interactions: List[Dict[str, Any]] | None = None
) -> Pipeline:
    """
    Build a preprocessing pipeline including standardization, one-hot-encoding
    and adding interaction terms.
    """

    num_steps = []
    if standardize:
        num_steps.append(("scaler", StandardScaler()))

    num_pipe = Pipeline(num_steps) if num_steps else "passthrough"

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", OneHotEncoder(
                drop="first", handle_unknown="ignore", sparse_output=False
            ), cat_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pre.set_output(transform="pandas")

    steps: List[Tuple[str, BaseEstimator]] = [("pre", pre)]

    interactions = interactions or []
    if any(i["status"] == "include" for i in interactions):
        steps.append(("inter", SelectedInteractions(interactions=interactions)))

    return Pipeline(steps)
