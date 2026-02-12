"""
Execution entry point of the ML project.
"""

from pathlib import Path
import shutil

import yaml

import pandas as pd

from src.loading import load_raw_data
from src.preparation import rename_features, numeric_to_string
from src.mappings import (
    FEATURE_NAME_MAPPING,
    FEATURE_CAT_MAPPING
)
from src.cross_validation import iter_cv_folds
from src.training import split_dataset
from src.logistic_regression import (
    fit_logistic_regression,
    evaluate_logistic_regression,
    predict_logistic_regression,
    store_logistic_regression
)
from src.neural_net import (
    fit_neuralnet,
    evaluate_neuralnet,
    predict_neuralnet,
    store_neural_net
)


with open("configs/run.yaml", "r") as f:
    RU_CONFIG = yaml.safe_load(f)
with open("configs/project.yaml", "r") as f:
    PR_CONFIG = yaml.safe_load(f)
with open("configs/cross_validation.yaml", "r") as f:
    CV_CONFIG = yaml.safe_load(f)
with open("configs/neural_net.yaml", "r") as f:
    NN_PARAMS = yaml.safe_load(f)
with open("configs/logistic_regression.yaml", "r") as f:
    LR_PARAMS = yaml.safe_load(f)


CV_CONFIG["n_splits"] = RU_CONFIG["cv_splits"]

MODEL_DICT = {
    "logistic_regression": {
        "params":       LR_PARAMS,
        "onehotencode": True,
        "fit":          fit_logistic_regression,
        "eval":         evaluate_logistic_regression,
        "pred":         predict_logistic_regression,
        "store":        store_logistic_regression
    },
    "neural_net": {
        "params":       NN_PARAMS,
        "onehotencode": True,
        "fit":          fit_neuralnet,
        "eval":         evaluate_neuralnet,
        "pred":         predict_neuralnet,
        "store":        store_neural_net
    }
}
PARAMS = MODEL_DICT[RU_CONFIG["model"]]["params"]
ONEHOTENCODE = MODEL_DICT[RU_CONFIG["model"]]["onehotencode"]


if __name__ == "__main__":
    """
    Execute ML pipeline.
    """

    # Prepare data

    ## Load variables
    dir_raw_data=PR_CONFIG["raw_data_dir"]
    raw_train_data=PR_CONFIG["raw_train_data"]
    raw_test_data=PR_CONFIG["raw_test_data"]
    label_col=PR_CONFIG["label_col"]

    ## Load raw data files
    train_data_raw = load_raw_data(dir_raw_data, raw_train_data)
    test_data_raw  = load_raw_data(dir_raw_data, raw_test_data)

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
    train_data_converted[label_col] = (
        train_data_converted[label_col].map(
            {"Absence": 0, "Presence": 1}
        ).astype(int)
    )

    ## Engineer and select features

    ### Drop feature 'id'
    train_data_prepared = train_data_converted.drop(columns=["id"])
    test_data_prepared  = test_data_converted.drop(columns=["id"])

    # Train model

    ## Create CV iterator
    cv_iterator = iter_cv_folds(train_data_prepared, label_col, CV_CONFIG)

    ## Train and evaluate model with CV

    model = RU_CONFIG["model"]

    cv_scores_list = []

    for fold_idx, train_df, val_df in cv_iterator:
        train_X, train_y = split_dataset(train_df, label_col)
        val_X, val_y = split_dataset(val_df, label_col)
        if ONEHOTENCODE:
            train_X = pd.get_dummies(train_X)
            val_X = (
                pd.get_dummies(val_X)
                .reindex(columns=train_X.columns, fill_value=0)
            )
        
        ml_model = MODEL_DICT[model]["fit"](train_X, train_y, PARAMS)

        ml_model_scores = MODEL_DICT[model]["eval"](ml_model, val_X, val_y)
        
        cv_scores_list.append(ml_model_scores)
    
    cv_scores = pd.concat(cv_scores_list, axis=0, ignore_index=True)

    cv_scores_table = cv_scores.describe()
    cv_scores_table.to_csv("output/cv_scores.csv")

    # Fit full model

    label_col=PR_CONFIG["label_col"]
    model = RU_CONFIG["model"]
    
    train_X, train_y = split_dataset(train_data_prepared, label_col)
    if ONEHOTENCODE:
        train_X = pd.get_dummies(train_X)

    full_ml_model = MODEL_DICT[model]["fit"](train_X, train_y, PARAMS)

    # Predict values for test data
    
    test_ids = test_data_converted["id"].to_numpy()

    test_X = test_data_prepared.copy()
    if ONEHOTENCODE:
        test_X = (
            pd.get_dummies(test_X)
            .reindex(columns=train_X.columns, fill_value=0)
        )
    
    y_proba = MODEL_DICT[model]["pred"](full_ml_model, test_X)
    
    output_df = pd.DataFrame(
        {
            "id": test_ids,
            "Heart Disease": y_proba,
        }
    )
        
    output_df.to_csv("output/predictions.csv", index=False)

    # Store model
    model = RU_CONFIG["model"]
    store_model = RU_CONFIG["store_model"]
    storage_name = RU_CONFIG["storage_name"]

    if store_model:
        dst_dir = Path("models", storage_name)
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        for path_name in [
            "configs",
            "output",
            "src"
        ]:
            src_path = Path(path_name)
            dst_path = Path(dst_dir, path_name)
            shutil.copytree(
                src=src_path,
                dst=dst_path,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
            )

        for file_name in [
            "run.py",
            "requirements.txt"
        ]:
            src_file = Path(file_name)
            dst_file = Path(dst_dir, file_name)
            dst_file.write_bytes(src_file.read_bytes())

        MODEL_DICT[model]["store"](full_ml_model, dst_dir, model)
