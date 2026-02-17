"""
Functions for engineering features.
"""

import pandas as pd


def standardize_feature(
    df: pd.DataFrame,
    column: str,
    mean_val: float | None = None,
    std_val: float | None = None
) -> tuple[pd.DataFrame, float, float]:
    """
    Standardize a numeric feature.
    """

    if mean_val is None:
        mean_ = df[column].mean()
    else:
        mean_ = mean_val
    
    if std_val is None:
        std_ = df[column].std()
    else:
        std_ = std_val

    df_scaled = df.copy()
    df_scaled[column] = (df_scaled[column] - mean_) / std_

    return df_scaled, mean_, std_
