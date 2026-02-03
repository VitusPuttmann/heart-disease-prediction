"""
Execution entry point of the ML project.
"""

from src.config import DIR_RAW_DATA
from src.loading import load_raw_data


RAW_TEST_DATA="test.csv"
RAW_TRAIN_DATA="train.csv"


if __name__ == "__main__":
    # Load raw data files
    test_data_raw = load_raw_data(DIR_RAW_DATA, RAW_TEST_DATA)
    train_data_raw = load_raw_data(DIR_RAW_DATA, RAW_TRAIN_DATA)
