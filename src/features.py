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


def add_polynomial(
    df: pd.DataFrame,
    feature_input: str,
    feature_output: str,
) -> pd.DataFrame:
    """
    Add a polynomial term of a numeric feature.
    """

    df_poly = df.copy()

    df_poly[feature_output] = df_poly[feature_input] ** 2

    return df_poly


def add_interaction_term(
    df: pd.DataFrame,
    feature_input_1: str,
    feature_input_2: str,
    feature_output: str
) -> pd.DataFrame:
    """
    Add an interaction term between two features.
    """

    df_intt = df.copy()
    
    df_intt[feature_output] = (
        df_intt[feature_input_1] * df_intt[feature_input_2]
    )

    return df_intt
