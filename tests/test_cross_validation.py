"""
Unit tests for CV splitter and iterator for cross-validation.
"""

from sklearn.model_selection import StratifiedKFold, KFold

from src.cross_validation import CVConfig, make_cv_splitter, iter_cv_folds


def test_make_cv_splitter_returns_expected_type():
    assert isinstance(make_cv_splitter(CVConfig(stratify=True)), StratifiedKFold)
    assert isinstance(make_cv_splitter(CVConfig(stratify=False)), KFold)


def test_iter_cv_folds_yields_correct_number_and_fold_idx_starts_at_1(raw_df):
    cfg = CVConfig(n_splits=5, shuffle=True, random_state=123456, stratify=True)
    folds = list(iter_cv_folds(raw_df, label_col="Heart Disease", cfg=cfg))

    assert len(folds) == cfg.n_splits
    assert folds[0][0] == 1
    assert folds[-1][0] == cfg.n_splits
