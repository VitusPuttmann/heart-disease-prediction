"""
Execution entry point of the ML project.
"""

import pandas as pd

from src.config import DIR_RAW_DATA
from src.loading import load_raw_data
from src.preparation import rename_features, numeric_to_string
from src.mappings import (
    FEATURE_NAME_MAPPING,
    FEATURE_CAT_MAPPING
)
from src.cross_validation import CVConfig, iter_cv_folds
from src.training import split_dataset
from src.models.xgboost_gbdt import (
    DC_PARAMS, fit_xgboost_gbdt, evaluate_xgboost_gbdt, predict_xgboost_gbdt
)


RAW_TRAIN_DATA = "train.csv"
RAW_TEST_DATA = "test.csv"

LABEL_COL = "heart_disease"

CV_CFG = CVConfig(
    n_splits=5,
    shuffle=True,
    random_state=483927,
    stratify=True
)

MODEL = "xgboost_gbdt"

MODEL_DICT = {
    "xgboost_gbdt": {
        "params": DC_PARAMS,
        "onehotencode": True,
        "fit": fit_xgboost_gbdt,
        "eval": evaluate_xgboost_gbdt,
        "pred": predict_xgboost_gbdt
    }
}

PARAMS = MODEL_DICT[MODEL]["params"]
ONEHOTENCODE = MODEL_DICT[MODEL]["onehotencode"]

if __name__ == "__main__":
    """
    Execute ML pipeline.
    """

    # Prepare data

    ## Load raw data files
    train_data_raw = load_raw_data(DIR_RAW_DATA, RAW_TRAIN_DATA)
    test_data_raw  = load_raw_data(DIR_RAW_DATA, RAW_TEST_DATA)

    ## Rename features
    train_data_renamed = rename_features(train_data_raw, FEATURE_NAME_MAPPING)
    test_data_renamed  = rename_features(test_data_raw, FEATURE_NAME_MAPPING)

    ## Convert features
    train_data_converted = train_data_renamed.copy()
    test_data_converted  = test_data_renamed.copy()

    ### Numeric to string for features
    for var_name, var_map in FEATURE_CAT_MAPPING.items():
        train_data_converted = numeric_to_string(
            train_data_converted, var_name, var_map
        )
        test_data_converted = numeric_to_string(
            test_data_converted, var_name, var_map
        )

    ### String to numeric for target
    train_data_converted[LABEL_COL] = (
        train_data_converted[LABEL_COL].map(
            {"Absence": 0, "Presence": 1}
        ).astype(int)
    )

    ## Engineer and select features

    ### Drop feature 'id'
    train_data_prepared = train_data_converted.drop(columns=["id"])
    test_data_prepared  = test_data_converted.drop(columns=["id"])

    # Train model

    ## Create CV iterator
    cv_iterator = iter_cv_folds(train_data_prepared, LABEL_COL, CV_CFG)

    ## Train and evaluate model with CV
    cv_scores_list = []

    for fold_idx, train_df, val_df in cv_iterator:
        train_X, train_y = split_dataset(train_df, LABEL_COL)
        val_X, val_y = split_dataset(val_df, LABEL_COL)
        if ONEHOTENCODE:
            train_X = pd.get_dummies(train_X)
            val_X = (
                pd.get_dummies(val_X)
                .reindex(columns=train_X.columns, fill_value=0)
            )
        
        ml_model = MODEL_DICT[MODEL]["fit"](train_X, train_y, PARAMS)

        ml_model_scores = MODEL_DICT[MODEL]["eval"](ml_model, val_X, val_y)
        
        cv_scores_list.append(ml_model_scores)
                
    cv_scores = pd.concat(cv_scores_list, axis=0, ignore_index=True)

    print(cv_scores.describe())     # For local inspection

    # Fit full model
    
    train_X, train_y = split_dataset(train_data_prepared, LABEL_COL)
    if ONEHOTENCODE:
        train_X = pd.get_dummies(train_X)

    full_ml_model = MODEL_DICT[MODEL]["fit"](train_X, train_y, PARAMS)

    # Predict values for test data
    
    test_ids = test_data_converted["id"].to_numpy()

    test_X = test_data_prepared.copy()
    if ONEHOTENCODE:
        test_X = (
            pd.get_dummies(test_X)
            .reindex(columns=train_X.columns, fill_value=0)
        )
    
    y_pred = MODEL_DICT[MODEL]["pred"](full_ml_model, test_X)
    
    output_df = pd.DataFrame(
        {
            "id": test_ids,
            "Heart Disease": y_pred,
        }
    )
        
    output_df.to_csv("output/predictions.csv", index=False)
