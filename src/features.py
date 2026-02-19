"""
Pipeline for engineering features.
"""

from __future__ import annotations

from typing import Optional, Any, Dict, List, Tuple, Literal

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    PowerTransformer,
    SplineTransformer
)


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        limits: Tuple[float, float] = (0.01, 0.99),
        columns: Optional[List[str]] = None,
    ):
        self.limits = limits
        self.columns = columns
        self._output_transform = None

    def fit(self, X: pd.DataFrame, y=None):
        lo, hi = self.limits

        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self._feature_names_in_ = list(X_df.columns)
        cols = self.columns or list(X_df.columns)
        self._cols_ = [c for c in cols if c in X_df.columns]

        self._bounds_: Dict[str, Tuple[float, float]] = {}
        for c in self._cols_:
            s = pd.to_numeric(X_df[c], errors="coerce")
            q_lo = float(s.quantile(lo))
            q_hi = float(s.quantile(hi))
            self._bounds_[c] = (q_lo, q_hi)

        return self

    def transform(self, X: pd.DataFrame):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_out = X_df.copy()
    
        for c in getattr(self, "_cols_", []):
            q_lo, q_hi = self._bounds_[c]
            s = pd.to_numeric(X_out[c], errors="coerce")
            clipped = s.clip(lower=q_lo, upper=q_hi)
            X_out[c] = clipped

        return X_out
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(getattr(self, "feature_names_in_", getattr(self, "_feature_names_in_", [])), dtype=object)
        return np.array(list(input_features), dtype=object)


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
            if spec["status"] != 1:
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


class HeartRateSTFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        age_col="age",
        max_hr_col="max_hr",
        st_col="st_depression",
        bp_col="blood_pressure",
        chol_col="cholesterol",
        fbs_col="rest_blood_sugar",
        exang_col="exercise_angina",
        vessels_col="number_vessels",
        thal_col="thallium",
        chol_threshold=240.0,
        chol_quantile=0.75,
    ):
        self.age_col = age_col
        self.max_hr_col = max_hr_col
        self.st_col = st_col
        self.bp_col = bp_col
        self.chol_col = chol_col
        self.fbs_col = fbs_col
        self.exang_col = exang_col
        self.vessels_col = vessels_col
        self.thal_col = thal_col
        self.chol_threshold = chol_threshold
        self.chol_quantile = chol_quantile

    def fit(self, X, y=None):
        self._added_cols_ = [
            "pred_max_hr", "hr_ratio", "hr_deficit", "st_per_hr", "st_x_hr_ratio",
            "high_bp", "chol_high", "fbs_high", "risk_count"
        ]

        if self.chol_threshold is None:
            s = pd.to_numeric(X[self.chol_col], errors="coerce")
            self._chol_thr_ = float(s.quantile(self.chol_quantile))
        else:
            self._chol_thr_ = float(self.chol_threshold)
        return self

    def transform(self, X):
        X_out = X.copy()

        pred_max_hr = 220.0 - X_out[self.age_col]
        denom = pred_max_hr.replace(0, pd.NA)

        X_out["pred_max_hr"] = pred_max_hr
        X_out["hr_ratio"] = X_out[self.max_hr_col] / denom
        X_out["hr_deficit"] = pred_max_hr - X_out[self.max_hr_col]

        X_out["st_per_hr"] = X_out[self.st_col] / (X_out[self.max_hr_col] + 1.0)
        X_out["st_x_hr_ratio"] = X_out[self.st_col] * X_out["hr_ratio"]
        X_out[
            ["hr_ratio", "st_x_hr_ratio"]
        ] = X_out[["hr_ratio", "st_x_hr_ratio"]].fillna(0.0)

        bp = pd.to_numeric(X_out[self.bp_col], errors="coerce")
        X_out["high_bp"] = (bp >= 140).astype(int)

        chol = pd.to_numeric(X_out[self.chol_col], errors="coerce")
        X_out["chol_high"] = (chol >= self._chol_thr_).astype(int)

        fbs = X_out[self.fbs_col]
        if pd.api.types.is_numeric_dtype(fbs):
            X_out["fbs_high"] = (fbs.astype(float) == 1.0).astype(int)
        else:
            X_out["fbs_high"] = fbs.astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)

        ex = X_out[self.exang_col]
        if pd.api.types.is_numeric_dtype(ex):
            exang_yes = (ex.astype(float) == 1.0).astype(int)
        else:
            exang_yes = ex.astype(str).str.lower().isin(["yes", "1", "true"]).astype(int)
        
        nv = X_out[self.vessels_col]
        if pd.api.types.is_numeric_dtype(nv):
            vessels_pos = (nv.astype(float) > 0.0).astype(int)
        else:
            vessels_pos = (~nv.astype(str).str.lower().isin(["zero", "0"])).astype(int)
        
        th = X_out[self.thal_col]
        if pd.api.types.is_numeric_dtype(th):
            th_abn = (th.astype(float) != 3.0).astype(int)
        else:
            th_abn = (~th.astype(str).str.lower().eq("normal")).astype(int)

        X_out["risk_count"] = (
            X_out["high_bp"]
            + X_out["chol_high"]
            + X_out["fbs_high"]
            + exang_yes
            + vessels_pos
            + th_abn
        ).astype(int)

        return X_out

    def get_feature_names_out(self, input_features=None):
        input_features = [] if input_features is None else list(input_features)
        return input_features + getattr(self, "_added_cols_", [])


class SelectedQuadratics(BaseEstimator, TransformerMixin):
    def __init__(self, features: List[str], suffix: str = "__sq"):
        self.features = features
        self.suffix = suffix

    def fit(self, X: pd.DataFrame, y=None):
        self._cols_ = [c for c in (self.features or []) if c in X.columns]
        self._added_cols_ = [f"{c}{self.suffix}" for c in self._cols_]
        return self

    def transform(self, X: pd.DataFrame):
        X_out = X.copy()
        for c in self._cols_:
            X_out[f"{c}{self.suffix}"] = X_out[c] ** 2
        return X_out

    def get_feature_names_out(self, input_features=None):
        input_features = [] if input_features is None else list(input_features)
        return input_features + getattr(self, "_added_cols_", [])


class SelectedCubicSplines(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        features: List[str],
        n_knots: int = 5,
        degree: int = 3,
        include_bias: bool = False,
        extrapolation: Literal[
            "error", "constant", "linear", "continue", "periodic"
        ] = "linear",
        suffix: str = "__cs"
    ):
        self.features = features
        self.n_knots = n_knots
        self.degree = degree
        self.include_bias = include_bias
        self.extrapolation = extrapolation
        self.suffix = suffix

    def fit(self, X: pd.DataFrame, y=None):
        self._cols_ = [c for c in (self.features or []) if c in X.columns]

        self._splines_ = {}
        self._added_cols_ = []

        for c in self._cols_:
            st = SplineTransformer(
                n_knots=self.n_knots,
                degree=self.degree,
                include_bias=self.include_bias,
                extrapolation=self.extrapolation,    # type: ignore[arg-type]
            )
            x = pd.to_numeric(X[c], errors="coerce").to_numpy().reshape(-1, 1)
            st.fit(x)
            self._splines_[c] = st

            n_out = st.transform(x[:1]).shape[1]
            self._added_cols_.extend([f"{c}{self.suffix}{j+1}" for j in range(n_out)])

        return self

    def transform(self, X: pd.DataFrame):
        X_out = X.copy()

        for c in self._cols_:
            st = self._splines_[c]
            x = pd.to_numeric(X_out[c], errors="coerce").to_numpy().reshape(-1, 1)

            basis = st.transform(x)
            n_out = basis.shape[1]

            for j in range(n_out):
                X_out[f"{c}{self.suffix}{j+1}"] = basis[:, j]

        return X_out

    def get_feature_names_out(self, input_features=None):
        input_features = [] if input_features is None else list(input_features)
        return input_features + getattr(self, "_added_cols_", [])


def build_preprocess_pipeline(
    num_cols: List[str],
    cat_cols: List[str],
    yj_cols: Optional[List[str]],
    standardize: bool,
    engineered_cols: List[str],
    interactions: List[Dict[str, Any]] | None = None,
    poly_features: Optional[List[str]] = None,
    spline_features: Optional[List[str]] = None, 
    winsor_cols: Optional[List[str]] = None
) -> Pipeline:
    """
    Build a preprocessing pipeline including (optional) winsorization,
    Yeo-Johnson transformation, standardization, one-hot-encoding,
    polynomials, splines, and interaction terms.
    """

    yj_cols = yj_cols or []
    yj_cols = [c for c in yj_cols if c in num_cols]
    other_num_cols = [c for c in num_cols if c not in set(yj_cols)]

    num_steps = []

    if winsor_cols is not None:
        winsor_cols = [c for c in winsor_cols if c in num_cols]
        if winsor_cols:
            num_steps.append(("winsor", Winsorizer(limits=(0.01, 0.99), columns=winsor_cols)))
    
    if yj_cols:
        yj_ct = ColumnTransformer(
            transformers=[
                ("yj", PowerTransformer(
                    method="yeo-johnson", standardize=False
                ), yj_cols),
                ("num_rest", "passthrough", other_num_cols)
            ],
            remainder="drop",
            verbose_feature_names_out=False
        )
        num_steps.append(("yeo_johnson", yj_ct))
    
    if standardize:
        num_steps.append(("scaler", StandardScaler()))

    if num_steps:
        num_pipe = Pipeline(num_steps)
    elif standardize:
        num_pipe = Pipeline([("scaler", StandardScaler())])
    else:
        num_pipe = "passthrough"

    if isinstance(num_pipe, tuple):
        num_pipe = Pipeline([num_pipe])

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

    steps: List[Tuple[str, BaseEstimator]] = []

    if engineered_cols:
        steps.append(
            ("hr_st_feats", HeartRateSTFeatures(
                age_col="age", max_hr_col="max_hr", st_col="st_depression"
            ))
        )

    steps.append(("pre", pre))
    
    poly_features = poly_features or []
    if poly_features:
        steps.append(("quad", SelectedQuadratics(features=poly_features)))
    
    spline_features = spline_features or []
    if spline_features:
        steps.append(("splines", SelectedCubicSplines(features=spline_features)))

    interactions = interactions or []
    if any(i["status"] == 1 for i in interactions):
        steps.append(("inter", SelectedInteractions(interactions=interactions)))

    return Pipeline(steps)
