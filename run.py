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
from src.cross_validation import CVConfig, iter_cv_folds


RAW_TRAIN_DATA="train.csv"
RAW_TEST_DATA="test.csv"

CFG=CVConfig(
    n_splits=5,
    shuffle=True,
    random_state=483927,
    stratify=True
)


if __name__ == "__main__":
    # Load raw data files
    train_data_raw = load_raw_data(DIR_RAW_DATA, RAW_TRAIN_DATA)
    test_data_raw = load_raw_data(DIR_RAW_DATA, RAW_TEST_DATA)

    # Rename features
    train_data_renamed = rename_features(train_data_raw, FEATURE_NAME_MAPPING)
    test_data_renamed = rename_features(test_data_raw, FEATURE_NAME_MAPPING)

    # Convert features
    train_data_converted= train_data_renamed.copy()
    test_data_converted = test_data_renamed.copy()

    for var_name, var_map in FEATURE_CAT_MAPPING.items():
        train_data_converted = numeric_to_string(train_data_converted, var_name, var_map)
        test_data_converted = numeric_to_string(test_data_converted, var_name, var_map)

    # Create CV iterator
    cv_iterator = iter_cv_folds(train_data_converted, "heart_disease", CFG)
