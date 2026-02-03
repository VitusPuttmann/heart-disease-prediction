"""
Test for loading raw data.
"""

import pandas as pd

from src.loading import load_raw_data


def test_load_raw_data_returns_nonempty_dataframe(raw_csv_path):
    df = load_raw_data(
        str(raw_csv_path.parent),
        str(raw_csv_path.name)
    )

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
