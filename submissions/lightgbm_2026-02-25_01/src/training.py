"""
Function for training the ML models.
"""

import pandas as pd


def split_dataset(
        df: pd.DataFrame,
        label_col: str
    ) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split features from target.
    """

    df_X = df.drop(columns=[label_col])
    df_y = df[label_col]
    
    return df_X, df_y
