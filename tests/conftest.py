"""
Configuration for the unit tests.
"""

from pathlib import Path

import yaml

import pytest

from src.loading import load_raw_data
from src.preparation import rename_features


with open("configs/features.yaml", "r") as f:
    FEATURES = yaml.safe_load(f)

feature_name_mapping = {
        v["name_raw"]: v["name_clean"] for v in FEATURES.values()}


@pytest.fixture(scope="module")
def raw_csv_path() -> Path:
    return Path(__file__).parent / "fixtures" / "raw_data_mock.csv"

@pytest.fixture(scope="module")
def raw_df(raw_csv_path):
    df_test = load_raw_data(
        str(raw_csv_path.parent),
        str(raw_csv_path.name)
    )

    return df_test

@pytest.fixture(scope="module")
def renamed_df(raw_df):
    df_test = rename_features(raw_df, feature_name_mapping)

    return df_test
