"""
Execution entry point of the ML project.
"""

from src.config import DIR_RAW_DATA
from src.loading import load_raw_data
from src.preparation import rename_features, numeric_to_string
from src.mappings import (
    FEATURE_NAME_MAPPING,
    FEATURE_CAT_MAPPING
)


RAW_TEST_DATA="test.csv"
RAW_TRAIN_DATA="train.csv"


if __name__ == "__main__":
    # Load raw data files
    test_data_raw = load_raw_data(DIR_RAW_DATA, RAW_TEST_DATA)
    train_data_raw = load_raw_data(DIR_RAW_DATA, RAW_TRAIN_DATA)

    # Rename features
    test_data_renamed = rename_features(test_data_raw, FEATURE_NAME_MAPPING)
    train_data_renamed = rename_features(train_data_raw, FEATURE_NAME_MAPPING)

    # Convert features
    test_data_converted = test_data_renamed.copy()
    train_data_converted= train_data_renamed.copy()

    for var_name, var_map in FEATURE_CAT_MAPPING.items():
        test_data_converted = numeric_to_string(test_data_converted, var_name, var_map)
        train_data_converted = numeric_to_string(train_data_converted, var_name, var_map)
