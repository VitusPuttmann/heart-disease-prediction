"""
CV splitter and iterator for cross-validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold


@dataclass(frozen=True)
class CVConfig:
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 483927
    stratify: bool = True


def make_cv_splitter(cfg: CVConfig):
    if cfg.stratify:
        return StratifiedKFold(
            n_splits=cfg.n_splits,
            shuffle=cfg.shuffle,
            random_state=cfg.random_state
        )
    
    return KFold(
        n_splits=cfg.n_splits,
        shuffle=cfg.shuffle,
        random_state=cfg.random_state
    )

def iter_cv_folds(
        df: pd.DataFrame,
        label_col: str,
        cfg: CVConfig
) -> Iterator[Tuple[int, pd.DataFrame, pd.DataFrame]]:
    y = df[label_col].to_numpy()

    splitter = make_cv_splitter(cfg=cfg)

    X_dummy = np.zeros(len(df))

    split_iter = (
        splitter.split(X_dummy, y) if cfg.stratify else splitter.split(X_dummy) # type: ignore
    )

    for fold_idx, (train_idx, val_idx) in enumerate(split_iter, start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        yield fold_idx, train_df, val_df
