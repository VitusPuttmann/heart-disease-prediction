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
            if spec.get("status", 0) != 1:
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
        X_out = X.copy()
        new_cols = {}

        for new_col, c1, c2 in getattr(self, "_plan_", []):
            a = pd.to_numeric(X_out[c1], errors="coerce")
            b = pd.to_numeric(X_out[c2], errors="coerce")
            new_cols[new_col] = (a * b).fillna(0.0)

        if new_cols:
            X_out = pd.concat(
                [X_out, pd.DataFrame(new_cols, index=X_out.index)], axis=1
            )

        return X_out

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
            "pred_max_hr",
            "hr_ratio",
            "hr_deficit",
            "st_per_hr",
            "st_x_hr_ratio",
            "high_bp",
            "chol_high",
            "fbs_high",
            "risk_count"
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
        new_cols = {}
        for c in self._cols_:
            s = pd.to_numeric(X_out[c], errors="coerce")
            new_cols[f"{c}{self.suffix}"] = (s ** 2).fillna(0.0)
        if new_cols:
            X_out = pd.concat([X_out, pd.DataFrame(new_cols, index=X_out.index)], axis=1)
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
        new_cols = {}

        for c in self._cols_:
            st = self._splines_[c]
            x = pd.to_numeric(X_out[c], errors="coerce").to_numpy().reshape(-1, 1)
            basis = st.transform(x)
            n_out = basis.shape[1]
            for j in range(n_out):
                new_cols[f"{c}{self.suffix}{j+1}"] = basis[:, j]

        if new_cols:
            X_out = pd.concat([X_out, pd.DataFrame(new_cols, index=X_out.index)], axis=1)
        return X_out

    def get_feature_names_out(self, input_features=None):
        input_features = [] if input_features is None else list(input_features)
        return input_features + getattr(self, "_added_cols_", [])


class TabularFeatureAugmenter(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        age_col: str = "age",
        sex_col: str = "sex",
        num_cols: List[str] | None = None,
        cat_cols: List[str] | None = None,
        te_cols: List[str] | None = None,
        te_pair_cols: List[Tuple[str, str]] | None = None,
        te_smoothing: float = 200.0,
        age_bin_width_10y: int = 10,
        age_bin_width_5y: int = 5,
    ):
        self.age_col = age_col
        self.sex_col = sex_col
        self.num_cols = num_cols or []
        self.cat_cols = cat_cols or []
        self.te_cols = te_cols or []
        self.te_pair_cols = te_pair_cols or []
        self.te_smoothing = te_smoothing
        self.age_bin_width_10y = age_bin_width_10y
        self.age_bin_width_5y = age_bin_width_5y

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        Xdf = X.copy()

        age = pd.to_numeric(Xdf[self.age_col], errors="coerce")
        a_min = int(np.floor(age.min() / 5.0) * 5)
        a_max = int(np.ceil(age.max() / 5.0) * 5)

        self._age_bins_10_ = np.arange(a_min, a_max + self.age_bin_width_10y, self.age_bin_width_10y)
        self._age_bins_5_  = np.arange(a_min, a_max + self.age_bin_width_5y,  self.age_bin_width_5y)

        # --- y handling (must be first; TE depends on it) ---
        if y is None:
            self._y_mean_ = 0.5
            y_ser = None
        else:
            y_ser = pd.Series(y).reset_index(drop=True)
            self._y_mean_ = float(y_ser.mean())

        X_tmp = Xdf.reset_index(drop=True)
        X_tmp["age_bin_10y"] = pd.cut(
            pd.to_numeric(X_tmp[self.age_col], errors="coerce"),
            bins=self._age_bins_10_,
            include_lowest=True,
        ).astype(str)
        X_tmp["age_bin_5y"] = pd.cut(
            pd.to_numeric(X_tmp[self.age_col], errors="coerce"),
            bins=self._age_bins_5_,
            include_lowest=True,
        ).astype(str)

        self._num_for_stats_ = [c for c in ["blood_pressure", "cholesterol", "max_hr", "st_depression", "age"] if c in X_tmp.columns]

        self._group_keys_ = [
            ["sex"],
            ["chest_pain"],
            ["exercise_angina"],
            ["thallium"],
            ["number_vessels"],
            ["ekg"],
            ["slope_st"],
            ["age_bin_10y", "sex"],
            ["thallium", "number_vessels"],
            ["sex", "chest_pain"],
            ["exercise_angina", "slope_st"],
        ]
        self._group_mean_maps_: dict[tuple[str, ...], pd.DataFrame] = {}
        self._group_count_maps_: dict[tuple[str, ...], pd.Series] = {}

        for g in self._group_keys_:
            g_tuple = tuple(g)
            if all(col in X_tmp.columns for col in g_tuple):
                grp = X_tmp.groupby(list(g_tuple), dropna=False)
                if self._num_for_stats_:
                    self._group_mean_maps_[g_tuple] = grp[self._num_for_stats_].mean()
                self._group_count_maps_[g_tuple] = grp.size()

        # --- target encodings ---
        self._te_maps_: dict[str, pd.Series] = {}
        self._te_pair_maps_: dict[tuple[str, str], pd.Series] = {}

        def _smooth_rate(n: pd.Series, m: pd.Series) -> pd.Series:
            a = float(self.te_smoothing)
            return (n * m + a * self._y_mean_) / (n + a)

        if y_ser is not None:
            for c in self.te_cols:
                if c not in X_tmp.columns:
                    continue
                # FORCE CONSISTENT KEYS (string) for mapping later
                grp = (
                    pd.DataFrame({c: X_tmp[c].astype(str), "_y": y_ser})
                    .groupby(c, dropna=False)["_y"]
                )
                self._te_maps_[c] = _smooth_rate(grp.size(), grp.mean())

            for c1, c2 in self.te_pair_cols:
                if c1 not in X_tmp.columns or c2 not in X_tmp.columns:
                    continue
                # FORCE CONSISTENT KEYS (string) for pair mapping later
                tmp = pd.DataFrame({c1: X_tmp[c1].astype(str), c2: X_tmp[c2].astype(str), "_y": y_ser})
                grp = tmp.groupby([c1, c2], dropna=False)["_y"]
                self._te_pair_maps_[(c1, c2)] = _smooth_rate(grp.size(), grp.mean())

        # --- percentile ranks ---
        self._pr_group_ = ("age_bin_10y", "sex")
        self._pr_cols_ = [c for c in ["blood_pressure", "cholesterol", "max_hr", "st_depression"] if c in X_tmp.columns]
        self._pr_sorted_: dict[tuple[Any, Any, str], np.ndarray] = {}
        if all(k in X_tmp.columns for k in self._pr_group_):
            for (ab, sx), sub in X_tmp.groupby(list(self._pr_group_), dropna=False):
                for c in self._pr_cols_:
                    arr = np.sort(pd.to_numeric(sub[c], errors="coerce").dropna().to_numpy())
                    self._pr_sorted_[(ab, sx, c)] = arr

        # --- COMPLETE, STATIC ADDED COL LIST (covers everything transform() can add) ---
        self._added_cols_ = [
            # bins
            "age_bin_10y",
            "age_bin_5y",

            # log1p + squares
            "log1p_cholesterol",
            "cholesterol_sq",
            "log1p_blood_pressure",
            "blood_pressure_sq",
            "log1p_st_depression",
            "st_depression_sq",
            "max_hr_sq",

            # ratios
            "cholesterol_over_blood_pressure",
            "blood_pressure_over_max_hr",
            "cholesterol_over_max_hr",
            "st_depression_over_max_hr",
            "max_hr_over_age",
            "blood_pressure_over_age",
            "cholesterol_over_age",
            "st_depression_over_age",

            # numeric interactions
            "age_x_max_hr",
            "age_x_st_depression",
            "max_hr_x_st_depression",
            "blood_pressure_x_st_depression",
            "cholesterol_x_st_depression",
            "blood_pressure_x_max_hr",
            "cholesterol_x_max_hr",

            # group counts (one per group key)
            "cnt_by_sex",
            "cnt_by_chest_pain",
            "cnt_by_exercise_angina",
            "cnt_by_thallium",
            "cnt_by_number_vessels",
            "cnt_by_ekg",
            "cnt_by_slope_st",
            "cnt_by_age_bin_10y_sex",
            "cnt_by_thallium_number_vessels",
            "cnt_by_sex_chest_pain",
            "cnt_by_exercise_angina_slope_st",

            # group means (one per group key x numeric stat variable)
            "mean_bp_by_sex",
            "mean_chol_by_sex",
            "mean_hr_by_sex",
            "mean_stdep_by_sex",
            "mean_age_by_sex",

            "mean_bp_by_chest_pain",
            "mean_chol_by_chest_pain",
            "mean_hr_by_chest_pain",
            "mean_stdep_by_chest_pain",
            "mean_age_by_chest_pain",

            "mean_bp_by_exercise_angina",
            "mean_chol_by_exercise_angina",
            "mean_hr_by_exercise_angina",
            "mean_stdep_by_exercise_angina",
            "mean_age_by_exercise_angina",

            "mean_bp_by_thallium",
            "mean_chol_by_thallium",
            "mean_hr_by_thallium",
            "mean_stdep_by_thallium",
            "mean_age_by_thallium",

            "mean_bp_by_number_vessels",
            "mean_chol_by_number_vessels",
            "mean_hr_by_number_vessels",
            "mean_stdep_by_number_vessels",
            "mean_age_by_number_vessels",

            "mean_bp_by_ekg",
            "mean_chol_by_ekg",
            "mean_hr_by_ekg",
            "mean_stdep_by_ekg",
            "mean_age_by_ekg",

            "mean_bp_by_slope_st",
            "mean_chol_by_slope_st",
            "mean_hr_by_slope_st",
            "mean_stdep_by_slope_st",
            "mean_age_by_slope_st",

            "mean_bp_by_age_bin_10y_sex",
            "mean_chol_by_age_bin_10y_sex",
            "mean_hr_by_age_bin_10y_sex",
            "mean_stdep_by_age_bin_10y_sex",
            "mean_age_by_age_bin_10y_sex",

            "mean_bp_by_thallium_number_vessels",
            "mean_chol_by_thallium_number_vessels",
            "mean_hr_by_thallium_number_vessels",
            "mean_stdep_by_thallium_number_vessels",
            "mean_age_by_thallium_number_vessels",

            "mean_bp_by_sex_chest_pain",
            "mean_chol_by_sex_chest_pain",
            "mean_hr_by_sex_chest_pain",
            "mean_stdep_by_sex_chest_pain",
            "mean_age_by_sex_chest_pain",

            "mean_bp_by_exercise_angina_slope_st",
            "mean_chol_by_exercise_angina_slope_st",
            "mean_hr_by_exercise_angina_slope_st",
            "mean_stdep_by_exercise_angina_slope_st",
            "mean_age_by_exercise_angina_slope_st",

            # distance-to-mean (only for age_bin_10y x sex)
            "bp_minus_mean_bp_by_agebin_sex",
            "chol_minus_mean_chol_by_agebin_sex",
            "hr_minus_mean_hr_by_agebin_sex",
            "stdep_minus_mean_stdep_by_agebin_sex",

            # percentile ranks within (age_bin_10y, sex)
            "pct_rank_bp_within_agebin_sex",
            "pct_rank_chol_within_agebin_sex",
            "pct_rank_hr_within_agebin_sex",
            "pct_rank_stdep_within_agebin_sex",
        ]

        # TE column names depend on configuration; add them deterministically from te_cols / te_pair_cols
        self._added_cols_ += [f"te_{c}" for c in self.te_cols]
        self._added_cols_ += [f"te_{c1}_{c2}" for (c1, c2) in self.te_pair_cols]

        return self

    def transform(self, X: pd.DataFrame):
        X_out = X.copy()
        new_cols: dict[str, Any] = {}

        # age bins
        age_num = pd.to_numeric(X_out[self.age_col], errors="coerce")
        new_cols["age_bin_10y"] = pd.cut(
            age_num, bins=self._age_bins_10_, include_lowest=True  # type: ignore
        ).astype(str)  # type: ignore
        new_cols["age_bin_5y"] = pd.cut(
            age_num, bins=self._age_bins_5_, include_lowest=True  # type: ignore
        ).astype(str)  # type: ignore

        # --- log1p + squares ---
        if "cholesterol" in X_out.columns:
            chol_num = pd.to_numeric(X_out["cholesterol"], errors="coerce")
            new_cols["log1p_cholesterol"] = np.log1p(chol_num)
            new_cols["cholesterol_sq"] = chol_num ** 2

        if "blood_pressure" in X_out.columns:
            bp_num = pd.to_numeric(X_out["blood_pressure"], errors="coerce")
            new_cols["log1p_blood_pressure"] = np.log1p(bp_num)
            new_cols["blood_pressure_sq"] = bp_num ** 2

        if "st_depression" in X_out.columns:
            st_num = pd.to_numeric(X_out["st_depression"], errors="coerce")
            new_cols["log1p_st_depression"] = np.log1p(st_num.clip(lower=0))
            new_cols["st_depression_sq"] = st_num ** 2

        if "max_hr" in X_out.columns:
            hr_num = pd.to_numeric(X_out["max_hr"], errors="coerce")
            new_cols["max_hr_sq"] = hr_num ** 2

        # --- ratios ---
        def _safe_div(a, b):
            b = pd.to_numeric(b, errors="coerce")
            return pd.to_numeric(a, errors="coerce") / (b.replace(0, np.nan))

        if {"cholesterol", "blood_pressure"}.issubset(X_out.columns):
            new_cols["cholesterol_over_blood_pressure"] = _safe_div(X_out["cholesterol"], X_out["blood_pressure"])
        if {"blood_pressure", "max_hr"}.issubset(X_out.columns):
            new_cols["blood_pressure_over_max_hr"] = _safe_div(X_out["blood_pressure"], X_out["max_hr"])
        if {"cholesterol", "max_hr"}.issubset(X_out.columns):
            new_cols["cholesterol_over_max_hr"] = _safe_div(X_out["cholesterol"], X_out["max_hr"])
        if {"st_depression", "max_hr"}.issubset(X_out.columns):
            new_cols["st_depression_over_max_hr"] = _safe_div(X_out["st_depression"], X_out["max_hr"])
        if {"max_hr", "age"}.issubset(X_out.columns):
            new_cols["max_hr_over_age"] = _safe_div(X_out["max_hr"], X_out["age"])
        if {"blood_pressure", "age"}.issubset(X_out.columns):
            new_cols["blood_pressure_over_age"] = _safe_div(X_out["blood_pressure"], X_out["age"])
        if {"cholesterol", "age"}.issubset(X_out.columns):
            new_cols["cholesterol_over_age"] = _safe_div(X_out["cholesterol"], X_out["age"])
        if {"st_depression", "age"}.issubset(X_out.columns):
            new_cols["st_depression_over_age"] = _safe_div(X_out["st_depression"], X_out["age"])

        # --- numeric interactions (small set) ---
        def _mul(a, b):
            return pd.to_numeric(a, errors="coerce") * pd.to_numeric(b, errors="coerce")

        if {"age", "max_hr"}.issubset(X_out.columns):
            new_cols["age_x_max_hr"] = _mul(X_out["age"], X_out["max_hr"])
        if {"age", "st_depression"}.issubset(X_out.columns):
            new_cols["age_x_st_depression"] = _mul(X_out["age"], X_out["st_depression"])
        if {"max_hr", "st_depression"}.issubset(X_out.columns):
            new_cols["max_hr_x_st_depression"] = _mul(X_out["max_hr"], X_out["st_depression"])
        if {"blood_pressure", "st_depression"}.issubset(X_out.columns):
            new_cols["blood_pressure_x_st_depression"] = _mul(X_out["blood_pressure"], X_out["st_depression"])
        if {"cholesterol", "st_depression"}.issubset(X_out.columns):
            new_cols["cholesterol_x_st_depression"] = _mul(X_out["cholesterol"], X_out["st_depression"])
        if {"blood_pressure", "max_hr"}.issubset(X_out.columns):
            new_cols["blood_pressure_x_max_hr"] = _mul(X_out["blood_pressure"], X_out["max_hr"])
        if {"cholesterol", "max_hr"}.issubset(X_out.columns):
            new_cols["cholesterol_x_max_hr"] = _mul(X_out["cholesterol"], X_out["max_hr"])

        # group stats helpers use updated bins (from new_cols) when present
        def _col(name: str):
            if name in new_cols:
                return new_cols[name]
            if name in X_out.columns:
                return X_out[name]
            # if a group key column is missing, return a NA series to avoid KeyError
            return pd.Series(pd.NA, index=X_out.index)

        def _merge_group_stats(g_tuple: tuple[str, ...]):
            # counts
            if g_tuple in self._group_count_maps_:
                cnt = self._group_count_maps_[g_tuple]
                name = "cnt_by_" + "_".join(g_tuple)
                frame = pd.DataFrame({k: _col(k) for k in g_tuple})
                new_cols[name] = pd.MultiIndex.from_frame(frame).map(cnt).fillna(0).astype(float) # type: ignore

            # means
            if g_tuple in self._group_mean_maps_:
                means = self._group_mean_maps_[g_tuple]
                frame = pd.DataFrame({k: _col(k) for k in g_tuple})
                idx = pd.MultiIndex.from_frame(frame)
                for c in self._num_for_stats_:
                    short = "bp" if c == "blood_pressure" else ("chol" if c == "cholesterol" else ("hr" if c == "max_hr" else ("stdep" if c == "st_depression" else c)))
                    name = f"mean_{short}_by_" + "_".join(g_tuple)
                    new_cols[name] = idx.map(means[c]).astype(float) # type: ignore

        for g in self._group_keys_:
            if all((col in X_out.columns) or (col in new_cols) for col in g):
                _merge_group_stats(tuple(g))

        # distance-to-mean for (age_bin_10y, sex)
        if {"age_bin_10y", "sex"}.issubset(set(X_out.columns) | set(new_cols.keys())):
            g = ("age_bin_10y", "sex")
            means = self._group_mean_maps_.get(g)
            if means is not None:
                frame = pd.DataFrame({"age_bin_10y": _col("age_bin_10y"), "sex": X_out["sex"]})
                idx = pd.MultiIndex.from_frame(frame)
                if "blood_pressure" in X_out.columns:
                    new_cols["bp_minus_mean_bp_by_agebin_sex"] = pd.to_numeric(X_out["blood_pressure"], errors="coerce") - idx.map(means["blood_pressure"]) # type: ignore # type: ignore
                if "cholesterol" in X_out.columns:
                    new_cols["chol_minus_mean_chol_by_agebin_sex"] = pd.to_numeric(X_out["cholesterol"], errors="coerce") - idx.map(means["cholesterol"]) # type: ignore
                if "max_hr" in X_out.columns:
                    new_cols["hr_minus_mean_hr_by_agebin_sex"] = pd.to_numeric(X_out["max_hr"], errors="coerce") - idx.map(means["max_hr"]) # type: ignore
                if "st_depression" in X_out.columns:
                    new_cols["stdep_minus_mean_stdep_by_agebin_sex"] = pd.to_numeric(X_out["st_depression"], errors="coerce") - idx.map(means["st_depression"]) # type: ignore

        # target encodings
        for c, mp in self._te_maps_.items():
            new_cols[f"te_{c}"] = X_out[c].astype(str).map(mp).fillna(self._y_mean_).astype(float)

        for (c1, c2), mp in self._te_pair_maps_.items():
            key = pd.MultiIndex.from_frame(X_out[[c1, c2]].astype(str))
            new_cols[f"te_{c1}_{c2}"] = key.map(mp).fillna(self._y_mean_).astype(float)

        # percentile ranks within (age_bin_10y, sex)
        if {"age_bin_10y", "sex"}.issubset(set(X_out.columns) | set(new_cols.keys())):
            ab = _col("age_bin_10y")
            sx = X_out["sex"]
            for c in self._pr_cols_:
                vals = pd.to_numeric(X_out[c], errors="coerce").to_numpy()
                out = np.full(shape=len(X_out), fill_value=np.nan, dtype=float)

                for i in range(len(X_out)):
                    if pd.isna(vals[i]):
                        continue
                    arr = self._pr_sorted_.get((ab.iloc[i], sx.iloc[i], c))
                    if arr is None or arr.size == 0:
                        continue
                    out[i] = np.searchsorted(arr, vals[i], side="right") / arr.size

                suffix = "bp" if c == "blood_pressure" else ("chol" if c == "cholesterol" else ("hr" if c == "max_hr" else "stdep"))
                new_cols[f"pct_rank_{suffix}_within_agebin_sex"] = out

        if new_cols:
            X_out = pd.concat([X_out, pd.DataFrame(new_cols, index=X_out.index)], axis=1)

        return X_out

    def get_feature_names_out(self, input_features=None):
            input_features = [] if input_features is None else list(input_features)
            return np.array(input_features + getattr(self, "_added_cols_", []), dtype=object)


class PostPreAugmenter(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        blocks: Optional[List[str]] = None,
        add_squares_for: Optional[List[str]] = None,
        add_log1p_for: Optional[List[str]] = None,
        add_ratios: Optional[List[Tuple[str, str, str]]] = None,
    ):
        self.blocks = blocks or []
        self.add_squares_for = add_squares_for or []
        self.add_log1p_for = add_log1p_for or []
        self.add_ratios = add_ratios or []

    def fit(self, X: pd.DataFrame, y=None):
        self._sq_cols_ = [c for c in self.add_squares_for if c in X.columns]
        self._log_cols_ = [c for c in self.add_log1p_for if c in X.columns]
        self._ratios_ = [(n, a, b) for (n, a, b) in self.add_ratios if a in X.columns and b in X.columns]
        return self

    def transform(self, X: pd.DataFrame):
        X_out = X.copy()
        new_cols: Dict[str, Any] = {}

        if "squares" in self.blocks:
            for c in self._sq_cols_:
                v = pd.to_numeric(X_out[c], errors="coerce")
                new_cols[f"{c}__sq"] = (v ** 2).fillna(0.0)

        if "log1p" in self.blocks:
            for c in self._log_cols_:
                v = pd.to_numeric(X_out[c], errors="coerce")
                new_cols[f"{c}__log1p"] = np.log1p(v.clip(lower=0)).fillna(0.0) # type: ignore

        if "ratios" in self.blocks:
            for name, num, den in self._ratios_:
                a = pd.to_numeric(X_out[num], errors="coerce")
                b = pd.to_numeric(X_out[den], errors="coerce").replace(0, np.nan)
                new_cols[name] = (a / b).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if new_cols:
            X_out = pd.concat([X_out, pd.DataFrame(new_cols, index=X_out.index)], axis=1)

        return X_out

    def get_feature_names_out(self, input_features=None):
        input_features = [] if input_features is None else list(input_features)

        out = list(input_features)
        if "squares" in self.blocks:
            out += [f"{c}__sq" for c in self._sq_cols_]
        if "log1p" in self.blocks:
            out += [f"{c}__log1p" for c in self._log_cols_]
        if "ratios" in self.blocks:
            out += [name for (name, _, _) in self._ratios_]
        return np.array(out, dtype=object)


def build_preprocess_pipeline(
    num_cols: List[str],
    cat_cols: List[str],
    yj_cols: Optional[List[str]],
    standardize: bool,
    one_hot: bool,
    augment: bool,
    engineered_cols: List[str],
    interactions: List[Dict[str, Any]] | None = None,
    poly_features: Optional[List[str]] = None,
    spline_features: Optional[List[str]] = None,
    winsor_cols: Optional[List[str]] = None,
    augment_blocks: Optional[List[str]] = None,
    augment_squares_for: Optional[List[str]] = None,
    augment_log1p_for: Optional[List[str]] = None,
    augment_ratios: Optional[List[Tuple[str, str, str]]] = None,
) -> Pipeline:
    
    tab_aug = TabularFeatureAugmenter(
        age_col="age",
        sex_col="sex",
        te_cols = ["chest_pain", "thallium", "ekg"],
        te_pair_cols = [
            ("sex", "chest_pain"),
            ("sex", "thallium"),
            ("chest_pain", "thallium")
        ]
    )
        
    # Columns created by TabularFeatureAugmenter
    tab_aug_cat_cols = ["age_bin_10y", "age_bin_5y"]
    tab_aug_num_cols = [
        "log1p_cholesterol", "cholesterol_sq",
        "log1p_blood_pressure", "blood_pressure_sq",
        "log1p_st_depression", "st_depression_sq",
        "max_hr_sq",
        "cholesterol_over_blood_pressure",
        "blood_pressure_over_max_hr",
        "cholesterol_over_max_hr",
        "st_depression_over_max_hr",
        "max_hr_over_age",
        "blood_pressure_over_age",
        "cholesterol_over_age",
        "st_depression_over_age",
        "age_x_max_hr", "age_x_st_depression",
        "max_hr_x_st_depression",
        "blood_pressure_x_st_depression",
        "cholesterol_x_st_depression",
        "blood_pressure_x_max_hr",
        "cholesterol_x_max_hr",
        # group counts (numeric)
        "cnt_by_sex",
        "cnt_by_chest_pain",
        "cnt_by_exercise_angina",
        "cnt_by_thallium",
        "cnt_by_number_vessels",
        "cnt_by_ekg",
        "cnt_by_slope_st",
        "cnt_by_age_bin_10y_sex",
        "cnt_by_thallium_number_vessels",
        "cnt_by_sex_chest_pain",
        "cnt_by_exercise_angina_slope_st",
        # group means (numeric)
        "mean_bp_by_sex", "mean_chol_by_sex", "mean_hr_by_sex", "mean_stdep_by_sex", "mean_age_by_sex",
        "mean_bp_by_chest_pain", "mean_chol_by_chest_pain", "mean_hr_by_chest_pain", "mean_stdep_by_chest_pain", "mean_age_by_chest_pain",
        "mean_bp_by_exercise_angina", "mean_chol_by_exercise_angina", "mean_hr_by_exercise_angina", "mean_stdep_by_exercise_angina", "mean_age_by_exercise_angina",
        "mean_bp_by_thallium", "mean_chol_by_thallium", "mean_hr_by_thallium", "mean_stdep_by_thallium", "mean_age_by_thallium",
        "mean_bp_by_number_vessels", "mean_chol_by_number_vessels", "mean_hr_by_number_vessels", "mean_stdep_by_number_vessels", "mean_age_by_number_vessels",
        "mean_bp_by_ekg", "mean_chol_by_ekg", "mean_hr_by_ekg", "mean_stdep_by_ekg", "mean_age_by_ekg",
        "mean_bp_by_slope_st", "mean_chol_by_slope_st", "mean_hr_by_slope_st", "mean_stdep_by_slope_st", "mean_age_by_slope_st",
        "mean_bp_by_age_bin_10y_sex", "mean_chol_by_age_bin_10y_sex", "mean_hr_by_age_bin_10y_sex", "mean_stdep_by_age_bin_10y_sex", "mean_age_by_age_bin_10y_sex",
        "mean_bp_by_thallium_number_vessels", "mean_chol_by_thallium_number_vessels", "mean_hr_by_thallium_number_vessels", "mean_stdep_by_thallium_number_vessels", "mean_age_by_thallium_number_vessels",
        "mean_bp_by_sex_chest_pain", "mean_chol_by_sex_chest_pain", "mean_hr_by_sex_chest_pain", "mean_stdep_by_sex_chest_pain", "mean_age_by_sex_chest_pain",
        "mean_bp_by_exercise_angina_slope_st", "mean_chol_by_exercise_angina_slope_st", "mean_hr_by_exercise_angina_slope_st", "mean_stdep_by_exercise_angina_slope_st", "mean_age_by_exercise_angina_slope_st",
        # distance-to-mean (numeric)
        "bp_minus_mean_bp_by_agebin_sex",
        "chol_minus_mean_chol_by_agebin_sex",
        "hr_minus_mean_hr_by_agebin_sex",
        "stdep_minus_mean_stdep_by_agebin_sex",
        # percentile ranks (numeric)
        "pct_rank_bp_within_agebin_sex",
        "pct_rank_chol_within_agebin_sex",
        "pct_rank_hr_within_agebin_sex",
        "pct_rank_stdep_within_agebin_sex",
    ]

    # TE outputs depend on config (here empty lists => none)
    tab_aug_num_cols += [f"te_{c}" for c in tab_aug.te_cols]
    tab_aug_num_cols += [f"te_{c1}_{c2}" for (c1, c2) in tab_aug.te_pair_cols]

    num_cols_pre = list(dict.fromkeys(num_cols + tab_aug_num_cols))
    cat_cols_pre = list(dict.fromkeys(cat_cols + tab_aug_cat_cols))

    yj_cols = (yj_cols or [])
    yj_cols = [c for c in yj_cols if c in num_cols_pre]
    other_num_cols = [c for c in num_cols_pre if c not in set(yj_cols)]

    num_steps = []

    if winsor_cols is not None:
        winsor_cols = [c for c in winsor_cols if c in num_cols_pre]
        if winsor_cols:
            num_steps.append(("winsor", Winsorizer(limits=(0.01, 0.99), columns=winsor_cols)))

    if yj_cols:
        yj_ct = ColumnTransformer(
            transformers=[
                ("yj", PowerTransformer(method="yeo-johnson", standardize=False), yj_cols),
                ("num_rest", "passthrough", other_num_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=True,
        )
        num_steps.append(("yeo_johnson", yj_ct))

    if standardize:
        num_steps.append(("scaler", StandardScaler()))

    num_pipe = Pipeline(num_steps) if num_steps else ("passthrough" if not standardize else Pipeline([("scaler", StandardScaler())]))

    cat_transformer = (
        OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
        if one_hot
        else "passthrough"
    )

    steps: List[Tuple[str, BaseEstimator]] = []
    steps.append(("tab_aug", tab_aug))

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols_pre),
            ("cat", cat_transformer, cat_cols_pre),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pre.set_output(transform="pandas")

    # (Optional) explicitly engineered columns that are already in `num_cols` / `cat_cols`
    # NOTE: any new columns created here must NOT be before `pre` unless you also pass them into `pre`.
    if engineered_cols:
        steps.append(("hr_st_feats", HeartRateSTFeatures(age_col="age", max_hr_col="max_hr", st_col="st_depression")))

    # --- REORDER: run `pre` early so nothing drops later features ---
    steps.append(("pre", pre))

    # --- NEW: augmentation AFTER pre (effective with remainder="drop") ---
    if augment:
        steps.append(
            ("post_aug", PostPreAugmenter(
                blocks=augment_blocks or [],
                add_squares_for=augment_squares_for or [],
                add_log1p_for=augment_log1p_for or [],
                add_ratios=augment_ratios or [],
            ))
        )

    # existing post-pre steps (unchanged)
    poly_features = poly_features or []
    if poly_features:
        steps.append(("quad", SelectedQuadratics(features=poly_features)))

    spline_features = spline_features or []
    if spline_features:
        steps.append(("splines", SelectedCubicSplines(features=spline_features)))

    interactions = interactions or []
    if any(i.get("status", 0) == 1 for i in interactions):
        steps.append(("inter", SelectedInteractions(interactions=interactions)))

    return Pipeline(steps)
