"""
Test for loading raw data.
"""

import pandas as pd

from src.loading import load_raw_data


def test_load_raw_data_returns_nonempty_dataframe(raw_csv_path):
    df_test = load_raw_data(
        str(raw_csv_path.parent),
        str(raw_csv_path.name)
    )

    assert isinstance(df_test, pd.DataFrame)
    assert not df_test.empty
