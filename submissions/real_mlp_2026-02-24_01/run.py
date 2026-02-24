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
from src.real_mlp import (
    fit_realmlp,
    evaluate_realmlp,
    predict_realmlp,
    store_realmlp
)
from src.xgboost import (
    fit_xgboost,
    evaluate_xgboost,
    predict_xgboost,
    store_xgboost
)
from src.ensemble import (
    evaluate_ensemble
)

with open("configs/run.yaml", "r") as f:
    RU_CONFIG = yaml.safe_load(f)
with open("configs/project.yaml", "r") as f:
    PR_CONFIG = yaml.safe_load(f)
with open("configs/cross_validation.yaml", "r") as f:
    CV_CONFIG = yaml.safe_load(f)
with open("configs/logistic_regression.yaml", "r") as f:
    LR_PARAMS = yaml.safe_load(f)
with open("configs/neural_net.yaml", "r") as f:
    NN_PARAMS = yaml.safe_load(f)
with open("configs/real_mlp.yaml", "r") as f:
    RM_PARAMS = yaml.safe_load(f)
with open("configs/xgboost.yaml", "r") as f:
    XG_PARAMS = yaml.safe_load(f)
with open("configs/features.yaml", "r") as f:
    FEATURES = yaml.safe_load(f)
FEATURE_KEYS = {
    k for k, v in FEATURES.items() if isinstance(v, dict) and "name_clean" in v
}

def is_engineered(k: str) -> bool:
    return isinstance(
        FEATURES.get(k), dict
    ) and FEATURES[k].get("source") == "engineered"

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
    },
    "real_mlp": {
        "params":       RM_PARAMS,
        "fit":          fit_realmlp,
        "eval":         evaluate_realmlp,
        "pred":         predict_realmlp,
        "store":        store_realmlp
    },
    "xgboost": {
        "params":       XG_PARAMS,
        "fit":          fit_xgboost,
        "eval":         evaluate_xgboost,
        "pred":         predict_xgboost,
        "store":        store_xgboost
    }
}

if __name__ == "__main__":
    """
    Execute ML pipeline.
    """

    # Obtain variables

    store_model = RU_CONFIG["store_model"]
    storage_name = RU_CONFIG["storage_name"]

    # Define output directories

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    if store_model:
        dst_dir = Path("models", storage_name)
        dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Measure runtime

    start = time.perf_counter()

    # Prepare data

    # Load variables
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
        feat["name_raw"]: feat["name_clean"]
        for feat in FEATURES.values()
        if isinstance(feat, dict) and "name_raw" in feat and "name_clean" in feat
    }

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
        if not isinstance(feat, dict):
            continue

        if feat.get("source") != "original":
            continue

        if feat.get(
            "type_raw"
        ) == "numeric" and feat.get("type_clean") == "categorical":
            var = feat["name_clean"]
            if var not in train_data_converted.columns:
                continue

            reversed_value_mapping = {v: k for k, v in feat["values"].items()}

            train_data_converted = numeric_to_string(
                train_data_converted, var, reversed_value_mapping
            )
            test_data_converted = numeric_to_string(
                test_data_converted, var, reversed_value_mapping
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
        if status == 1
        and key in FEATURE_KEYS
        and not is_engineered(key)
    ]

    engineered_cols = [
        FEATURES[key]["name_clean"]
        for key, status in RU_CONFIG["features"].items()
        if status == 1
        and key in FEATURE_KEYS
        and is_engineered(key)
    ]

    num_cols = [
        FEATURES[key]["name_clean"]
        for key, status in RU_CONFIG["features"].items()
        if status == 1
        and key in FEATURE_KEYS
        and FEATURES[key]["type_clean"] == "numeric"
    ]

    winsor_features = [
        FEATURES[key]["name_clean"]
        for key, status in RU_CONFIG["winsor_features"].items()
        if status == 1
        and key in FEATURE_KEYS
    ]

    cat_cols = [
        FEATURES[key]["name_clean"]
        for key, status in RU_CONFIG["features"].items()
        if status == 1
        and key in FEATURE_KEYS
        and FEATURES[key]["type_clean"] == "categorical"
        and not is_engineered(key)
    ]

    yj_cols = [
        FEATURES[key]["name_clean"] if key in FEATURE_KEYS else key
        for key, value in RU_CONFIG["yeojohnson"].items()
        if value == 1
    ]
    yj_cols = [c for c in yj_cols if c in num_cols]

    interactions = [
        {**i, "status": 1}
        for i in FEATURES["interactions"]
        if RU_CONFIG.get(i["name"], 0) == 1
    ]

    poly_features = [
        FEATURES[key]["name_clean"] if key in FEATURE_KEYS else key
        for key, value in RU_CONFIG["poly_features"].items()
        if value == 1
    ]
    poly_features = [c for c in poly_features if c in num_cols]

    spline_features = [
        FEATURES[key]["name_clean"] if key in FEATURE_KEYS else key
        for key, value in RU_CONFIG["spline_features"].items()
        if value == 1
    ]
    spline_features = [c for c in spline_features if c in num_cols]

    # Select features

    train_data_prepared = train_data_converted[
        selected_feature_columns + [label_col]
    ]
    test_data_prepared  = test_data_converted[selected_feature_columns]

    # Train model

    ## Initiate output

    cv_scores_list = []

    full_pred = pd.DataFrame()

    output_df = pd.DataFrame()

    ## Create model loop

    for model in RU_CONFIG["models"]:
        
        ## Obtain parameters

        PARAMS = MODEL_DICT[model]["params"]

        ## Create CV iterator
        
        cv_iterator = iter_cv_folds(train_data_prepared, label_col, CV_CONFIG)

        ## Train and evaluate model with CV

        for fold_idx, train_df, val_df in cv_iterator:
            train_X, train_y = split_dataset(train_df, label_col)
            val_X, val_y = split_dataset(val_df, label_col)
            
            pre_pipe = build_preprocess_pipeline(
                num_cols=num_cols,
                cat_cols=cat_cols,
                yj_cols=yj_cols,
                standardize=bool(RU_CONFIG["standardize"]),
                engineered_cols=engineered_cols,
                interactions=interactions,
                poly_features=poly_features,
                winsor_cols=winsor_features
            )

            train_Xp = pre_pipe.fit_transform(train_X)
            val_Xp   = pre_pipe.transform(val_X)

            ml_model = MODEL_DICT[model]["fit"](train_Xp, train_y, PARAMS)

            ml_model_scores = MODEL_DICT[model]["eval"](ml_model, val_Xp, val_y)
            
            cv_scores_list.append(ml_model_scores)

        # Fit full model

        if RU_CONFIG["fit_model"]:
        
            label_col=PR_CONFIG["label_col"]
            
            train_X, train_y = split_dataset(train_data_prepared, label_col)

            pre_pipe = build_preprocess_pipeline(
                num_cols=num_cols,
                cat_cols=cat_cols,
                yj_cols=yj_cols,
                standardize=bool(RU_CONFIG["standardize"]),
                engineered_cols=engineered_cols,
                interactions=interactions,
                poly_features=poly_features,
                winsor_cols=winsor_features
            )

            train_Xp = pre_pipe.fit_transform(train_X)
            
            full_ml_model = MODEL_DICT[model]["fit"](train_Xp, train_y, PARAMS)

            # Store evaluation scores

            y_proba_full = MODEL_DICT[model]["pred"](full_ml_model, train_Xp)
            
            full_pred[model] = y_proba_full

            # Store regression table

            if (
                model == "logistic_regression" and
                RU_CONFIG["store_regression"]
            ):
                store_regression_table(train_Xp, train_y, output_dir) # type: ignore

            # Store model

            if store_model:
                MODEL_DICT[model]["store"](full_ml_model, dst_dir, model) # type: ignore

            # Predict values for test data
            
            test_ids = test_data_converted["id"].to_numpy()

            test_X = test_data_prepared

            test_Xp = pre_pipe.transform(test_X)
            
            y_proba = MODEL_DICT[model]["pred"](full_ml_model, test_Xp)
            
            col_name_1 = "id" + model
            col_name_2 = "y_proba" + model

            output_df[col_name_1] = test_ids
            output_df[col_name_2] = y_proba
    
    # Store outputs

    ##  CV scores

    cv_scores = pd.concat(cv_scores_list, axis=0, ignore_index=True)
    cv_scores_table = cv_scores.describe().round(5)
    cv_scores_table.to_csv(output_dir / "cv_scores.csv", index=False)

    ## Full model scores

    if RU_CONFIG["fit_model"]:
        full_pred["combined"] = 0
        num_models = 0
        for model in RU_CONFIG["models"]:
            num_models += 1
            full_pred["combined"] += full_pred[model]
        full_pred["combined"] = full_pred["combined"] / num_models

        y_true = train_y
        y_proba = full_pred["combined"]
        full_score = evaluate_ensemble(y_true, y_proba)

        with open(output_dir / "full_score.csv", "w") as f:
            f.write(f"{full_score:.5f}\n")
    
    ## Predicted probabilities

    output_df["final_predictions"] = 0
    num_models = 0
    for model in RU_CONFIG["models"]:
        num_models += 1
        col_name = "y_proba" + model
        output_df["final_predictions"] += output_df[col_name]
    output_df["final_predictions"] = output_df["final_predictions"] / num_models
    output_df.to_csv(output_dir / "predictions.csv", index=False)

    ## Models

    if store_model:
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
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                dirs_exist_ok=True
            )

        for file_name in [
            "run.py",
            "requirements.txt"
        ]:
            src_file = Path(file_name)
            dst_file = Path(dst_dir, file_name)
            dst_file.write_bytes(src_file.read_bytes())

    # Calculate runtime

    end = time.perf_counter()
    print(f"Runtime: {(end - start) / 60:.1f} m")
