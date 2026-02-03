"""
Test for preparing raw data.
"""

import pandas as pd

from src.preparation import rename_features, numeric_to_string
from src.mappings import (
    FEATURE_NAME_MAPPING,
    FEATURE_CAT_MAPPING
)

NEW_FEAT_NAMES = [
    "id",
    "age",
    "sex",
    "chest_pain_type",
    "blood_pressure",
    "cholesterol",
    "fbs_over_120",
    "ekg_results",
    "max_hr",
    "exercise_angina",
    "st_depression",
    "slope_st",
    "number_vessels_fluro",
    "thallium",
    "heart_disease"
]


def test_rename_features_returns_dataframe(raw_df):
    df_test = rename_features(raw_df, FEATURE_NAME_MAPPING)

    assert isinstance(df_test, pd.DataFrame)


def test_rename_features_renames_features(raw_df):
    df_test = rename_features(raw_df, FEATURE_NAME_MAPPING)
    
    assert list(df_test.columns) == NEW_FEAT_NAMES


def test_numeric_to_string_converts_features(renamed_df):
    df_test = renamed_df.copy()

    for var_name, var_map in FEATURE_CAT_MAPPING.items():
        df_test = numeric_to_string(df_test, var_name, var_map)

    for var_name, var_map in FEATURE_CAT_MAPPING.items():
        assert sorted(df_test[var_name].unique()) == sorted(list(var_map.values()))
