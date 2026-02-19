"""
Execution entry point of the ML project.
"""

import time

from pathlib import Path
import shutil

import yaml

import pandas as pd

from src.loading import load_raw_data
from src.preparation import rename_features, numeric_to_string
from src.features import (
    build_preprocess_pipeline
)
from src.cross_validation import iter_cv_folds
from src.training import split_dataset
from src.logistic_regression import (
    store_regression_table,
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
with open("configs/features.yaml", "r") as f:
    FEATURES = yaml.safe_load(f)

CV_CONFIG["n_splits"] = RU_CONFIG["cv_splits"]

MODEL_DICT = {
    "logistic_regression": {
        "params":       LR_PARAMS,
        "fit":          fit_logistic_regression,
        "eval":         evaluate_logistic_regression,
        "pred":         predict_logistic_regression,
        "store":        store_logistic_regression
    },
    "neural_net": {
        "params":       NN_PARAMS,
        "fit":          fit_neuralnet,
        "eval":         evaluate_neuralnet,
        "pred":         predict_neuralnet,
        "store":        store_neural_net
    }
}

PARAMS = MODEL_DICT[RU_CONFIG["model"]]["params"]


if __name__ == "__main__":
    """
    Execute ML pipeline.
    """

    # Define output directory

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Measure runtime

    start = time.perf_counter()

    # Prepare data

    ## Load variables
    dir_raw_data=PR_CONFIG["raw_data_dir"]
    raw_train_data=PR_CONFIG["raw_train_data"]
    raw_test_data=PR_CONFIG["raw_test_data"]
    label_col=PR_CONFIG["label_col"]

    ## Load raw data files
    train_data_raw = load_raw_data(dir_raw_data, raw_train_data)
    test_data_raw  = load_raw_data(dir_raw_data, raw_test_data)

    ## Rename target
    train_data_renamed = train_data_raw.rename(
        columns={"Heart Disease": "heart_disease"}
    )
    test_data_renamed  = test_data_raw.rename(
        columns={"Heart Disease": "heart_disease"}
    )

    ## Rename features
    feature_name_mapping = {
        feat["name_raw"]: feat["name_clean"] for feat in FEATURES.values()}
    train_data_renamed = rename_features(
        train_data_renamed, feature_name_mapping
    )
    test_data_renamed = rename_features(
        test_data_renamed, feature_name_mapping
    )

    ## Convert features
    train_data_converted = train_data_renamed.copy()
    test_data_converted  = test_data_renamed.copy()

    ### Numeric to string for features
    for feat in FEATURES.values():
        if feat["type_raw"] == "numeric" and feat["type_clean"] == "categorical":
            reversed_value_mapping = {v: k for k, v in feat["values"].items()}
            train_data_converted = numeric_to_string(
                train_data_converted, feat["name_clean"], reversed_value_mapping
            )
            test_data_converted = numeric_to_string(
                test_data_converted, feat["name_clean"], reversed_value_mapping
            )

    ### String to numeric for target
    train_data_converted[label_col] = (
        train_data_converted[label_col].map(
            {"Absence": 0, "Presence": 1}
        ).astype(int)
    )

    # Prepare data preprocessing pipeline

    selected_feature_columns = [
       FEATURES[key]["name_clean"]
        for key, status in RU_CONFIG["features"].items()
        if status == "include"
    ]

    num_cols = [
        FEATURES[key]["name_clean"]
        for key, status in RU_CONFIG["features"].items()
        if status == "include"
        and FEATURES[key]["type_clean"] == "numeric"
    ]

    cat_cols = [
        FEATURES[key]["name_clean"]
        for key, status in RU_CONFIG["features"].items()
        if status == "include"
        and FEATURES[key]["type_clean"] == "categorical"
    ]

    interactions  = RU_CONFIG["interactions"]
    poly_features = RU_CONFIG["poly_features"]

    # Select features

    train_data_prepared = train_data_converted[
        selected_feature_columns + [label_col]
    ]
    test_data_prepared  = test_data_converted[selected_feature_columns]

    # Train model

    ## Create CV iterator
    
    cv_iterator = iter_cv_folds(train_data_prepared, label_col, CV_CONFIG)

    ## Train and evaluate model with CV

    model = RU_CONFIG["model"]

    cv_scores_list = []

    for fold_idx, train_df, val_df in cv_iterator:
        train_X, train_y = split_dataset(train_df, label_col)
        val_X, val_y = split_dataset(val_df, label_col)
        
        pre_pipe = build_preprocess_pipeline(
            num_cols=num_cols,
            cat_cols=cat_cols,
            standardize=bool(RU_CONFIG["standardize"]),
            interactions=interactions,
            poly_features=poly_features
        )

        train_Xp = pre_pipe.fit_transform(train_X)
        val_Xp   = pre_pipe.transform(val_X)

        ml_model = MODEL_DICT[model]["fit"](train_Xp, train_y, PARAMS)

        ml_model_scores = MODEL_DICT[model]["eval"](ml_model, val_Xp, val_y)
        
        cv_scores_list.append(ml_model_scores)
    
    cv_scores = pd.concat(cv_scores_list, axis=0, ignore_index=True)

    cv_scores_table = cv_scores.describe().round(5)

    cv_scores_table.to_csv(output_dir / "cv_scores.csv", index=False)

    # Fit full model

    label_col=PR_CONFIG["label_col"]
    model = RU_CONFIG["model"]
    
    train_X, train_y = split_dataset(train_data_prepared, label_col)

    pre_pipe = build_preprocess_pipeline(
        num_cols=num_cols,
        cat_cols=cat_cols,
        standardize=bool(RU_CONFIG["standardize"]),
        interactions=interactions,
        poly_features=poly_features
    )

    train_Xp = pre_pipe.fit_transform(train_X)
    
    full_ml_model = MODEL_DICT[model]["fit"](train_Xp, train_y, PARAMS)

    if RU_CONFIG["model"] == "logistic_regression":
        store_regression_table(train_Xp, train_y, output_dir)

    # Predict values for test data
    
    test_ids = test_data_converted["id"].to_numpy()

    test_X = test_data_prepared[selected_feature_columns]

    test_Xp = pre_pipe.transform(test_X)
    
    y_proba = MODEL_DICT[model]["pred"](full_ml_model, test_Xp)
    
    output_df = pd.DataFrame(
        {
            "id": test_ids,
            "Heart Disease": y_proba,
        }
    )
        
    output_df.to_csv(output_dir / "predictions.csv", index=False)

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
        
    end = time.perf_counter()
    print(f"Runtime: {(end - start) / 60:.1f} m")
