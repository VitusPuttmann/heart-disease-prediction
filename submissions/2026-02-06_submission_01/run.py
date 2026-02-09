"""
Execution entry point of the ML project.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import DIR_RAW_DATA
from src.loading import load_raw_data
from src.preparation import rename_features, numeric_to_string
from src.mappings import (
    FEATURE_NAME_MAPPING,
    FEATURE_CAT_MAPPING
)
from src.cross_validation import CVConfig, iter_cv_folds
from src.model import fit_and_evaluate_decision_tree


RAW_TRAIN_DATA="train.csv"
RAW_TEST_DATA="test.csv"

LABEL_COL = "heart_disease"

CFG=CVConfig(
    n_splits=5,
    shuffle=True,
    random_state=483927,
    stratify=True
)


if __name__ == "__main__":
    # Load raw data files
    train_data_raw = load_raw_data(DIR_RAW_DATA, RAW_TRAIN_DATA)
    test_data_raw = load_raw_data(DIR_RAW_DATA, RAW_TEST_DATA)

    # Rename features
    train_data_renamed = rename_features(train_data_raw, FEATURE_NAME_MAPPING)
    test_data_renamed = rename_features(test_data_raw, FEATURE_NAME_MAPPING)

    # Convert features
    train_data_converted = train_data_renamed.copy()
    test_data_converted = test_data_renamed.copy()

    for var_name, var_map in FEATURE_CAT_MAPPING.items():
        train_data_converted = numeric_to_string(
            train_data_converted, var_name, var_map
        )
        test_data_converted = numeric_to_string(
            test_data_converted, var_name, var_map
        )

    train_data_converted[LABEL_COL] = (
        train_data_converted[LABEL_COL].map(
            {"Absence": 0, "Presence": 1}
        ).astype(int)
    )

    # Drop ID
    train_data_converted = train_data_converted.drop(columns=["id"])
    
    # Create CV iterator
    cv_iterator = iter_cv_folds(train_data_converted, "heart_disease", CFG)

    # Train and evaluate model with CV
    cv_scores = []
    for fold_idx, train_df, val_df in cv_iterator:
        model, score = fit_and_evaluate_decision_tree(
            train_df,
            val_df,
            LABEL_COL,
            evaluate=True)
        cv_scores.append(score)

    mean = np.mean(cv_scores)
    print(mean)    # LOCAL DEVELOPMENT ONLY - REMOVE
    std  = np.std(cv_scores, ddof=1)
    print(std)    # LOCAL DEVELOPMENT ONLY - REMOVE

    # Fit full model
    full_model, train_columns = fit_and_evaluate_decision_tree(
        train_data_converted,
        None,
        label_col=LABEL_COL,
        evaluate=False
    )

    # Predict values for test data
    test_ids = test_data_converted["id"].to_numpy()
    test_X = test_data_converted.drop(columns=["id"])
    test_X = pd.get_dummies(test_X, dummy_na=True)
    test_X = test_X.reindex(columns=train_columns, fill_value=0)
    dX = xgb.DMatrix(test_X)

    proba = full_model.predict(dX)
    y_pred = (proba >= 0.5).astype(int)

    output_df = pd.DataFrame(
        {
            "id": test_ids,
            "Heart Disease": y_pred,
        }
    )

    output_df.to_csv("output/predictions.csv", index=False)
