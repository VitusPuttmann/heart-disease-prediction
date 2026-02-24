"""
Function for importing raw data.
"""

from pathlib import Path

import pandas as pd


def load_raw_data(dir_raw_data: str, name_data_file: str) -> pd.DataFrame:
    path_data_file = Path(dir_raw_data) / name_data_file

    if not path_data_file.exists():
        raise FileNotFoundError(f"Raw data file not found: {path_data_file}")
    
    return pd.read_csv(path_data_file)
