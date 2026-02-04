"""
Functions for preparing the raw data.
"""

import pandas as pd


def rename_features(df_input: pd.DataFrame, name_mapping: dict) -> pd.DataFrame:
    df_output = df_input.copy()

    df_output = df_output.rename(columns=name_mapping)

    return df_output

def numeric_to_string(
        df_input: pd.DataFrame, var: str, trans_dict: dict
    ) -> pd.DataFrame:
    df_output = df_input.copy()

    df_output[var] = df_output[var].map(trans_dict)

    return df_output
