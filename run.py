"""
Execution entry point of the ML project.
"""

from pathlib import Path
import shutil
import time

import pandas as pd
from pandas.api.types import is_numeric_dtype

from configs.loader import load_yaml
from src.loading import load_raw_data
from src.preparation import rename_features, numeric_to_string
from src.features import build_preprocess_pipeline
from src.cross_validation import iter_cv_folds
from src.training import split_dataset
from src.models.catboost import (
    fit_catboost,
    evaluate_catboost,
    predict_catboost,
    cat_feature_indices_after_preprocessing,
)
from src.models.registry import build_model_registry
from src.ensemble import evaluate_ensemble


RU_CONFIG = load_yaml("configs/run.yaml")
PR_CONFIG = load_yaml("configs/project.yaml")
CV_CONFIG = load_yaml("configs/cross_validation.yaml")
FEATURES  = load_yaml("configs/features.yaml")
CB_PARAMS = load_yaml("params/catboost.yaml")
LG_PARAMS = load_yaml("params/lightgbm.yaml")
LR_PARAMS = load_yaml("params/logistic_regression.yaml")
NN_PARAMS = load_yaml("params/neural_net.yaml")
RM_PARAMS = load_yaml("params/real_mlp.yaml")
XG_PARAMS = load_yaml("params/xgboost.yaml")

CV_CONFIG["n_splits"] = RU_CONFIG["cv_splits"]

MODEL_REG = build_model_registry(
    LG_PARAMS,
    CB_PARAMS,
    LR_PARAMS,
    NN_PARAMS,
    RM_PARAMS,
    XG_PARAMS,
)

FEATURE_KEYS = {
    k for k, v in FEATURES.items() if isinstance(v, dict) and "name_clean" in v
}

def is_engineered(k: str) -> bool:
    return isinstance(
        FEATURES.get(k), dict
    ) and FEATURES[k].get("source") == "engineered"

def build_drop_features(columns: list[str], cfg: dict) -> list[str]:
    disabled = {k for k, v in cfg.items() if int(v) == 0}
    drop_features: list[str] = []

    for col in columns:
        tail = col.split("__")[-1]

        if "age_x_exercise_angina" in disabled and "age_x_exercise_angina__" in col:
            drop_features.append(col)
            continue
        if "age_x_number_vessels" in disabled and "age_x_number_vessels__" in col:
            drop_features.append(col)
            continue
        if "age_x_thallium" in disabled and "age_x_thallium__" in col:
            drop_features.append(col)
            continue

        if "age_bin_10y" in disabled and (
            tail == "age_bin_10y" or tail.startswith("age_bin_10y_")
        ):
            drop_features.append(col)
            continue
        if "age_bin_5y" in disabled and (
            tail == "age_bin_5y" or tail.startswith("age_bin_5y_")
        ):
            drop_features.append(col)
            continue

        if tail in disabled:
            drop_features.append(col)

    return drop_features


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

    additional_features_cfg = RU_CONFIG.get("additional_features", {})

    # Select features

    train_data_prepared = train_data_converted[
        selected_feature_columns + [label_col]
    ]
    test_data_prepared  = test_data_converted[selected_feature_columns]

    # Train model

    ## Initiate output

    cv_scores_list = []
    oof_pred_all = pd.DataFrame(index=train_data_prepared.index)
    output_df = pd.DataFrame()

    ## Create model loop

    for model in RU_CONFIG["models"]:
        
        ## Obtain parameters

        PARAMS = MODEL_REG[model].params

        ## Create CV iterator
        
        cv_iterator = iter_cv_folds(train_data_prepared, label_col, CV_CONFIG)

        ## Train and evaluate model with CV

        oof_proba = pd.Series(index=train_data_prepared.index, dtype=float)

        for _, train_df, val_df in cv_iterator:
            train_X, train_y = split_dataset(train_df, label_col)
            val_X, val_y = split_dataset(val_df, label_col)
            
            pre_pipe = build_preprocess_pipeline(
                num_cols=num_cols,
                cat_cols=cat_cols,
                yj_cols=yj_cols,
                standardize=bool(RU_CONFIG["standardize"]),
                one_hot=PARAMS["one_hot"],
                augment=PARAMS["augment"],
                engineered_cols=engineered_cols,
                interactions=interactions,
                poly_features=poly_features,
                winsor_cols=winsor_features,
                augment_blocks=["squares", "log1p", "ratios"],
                augment_squares_for=[
                    "cholesterol",
                    "blood_pressure",
                    "st_depression",
                    "max_hr"
                ],
                augment_log1p_for=[
                    "cholesterol",
                    "blood_pressure",
                    "st_depression"
                ],
                augment_ratios=[
                    ("cholesterol_over_blood_pressure",
                     "cholesterol",
                     "blood_pressure"),
                    ("st_depression_over_max_hr",
                     "st_depression",
                     "max_hr"),
                    ("max_hr_over_age",
                     "max_hr",
                     "age"),
                ]
            )
            
            train_Xp = pre_pipe.fit_transform(train_X, train_y)
            val_Xp   = pre_pipe.transform(val_X)

            drop_features = build_drop_features(
                list(train_Xp.columns), additional_features_cfg
            )

            train_Xp = train_Xp.drop(
               columns=[c for c in drop_features if c in train_Xp.columns]
            )
            val_Xp = val_Xp.drop(
               columns=[c for c in drop_features if c in val_Xp.columns]
            )
            val_Xp = val_Xp.reindex(columns=train_Xp.columns, fill_value=0.0)

            if model == "catboost" and not PARAMS["one_hot"]:
                interval_cols = [
                    c for c in train_Xp.columns
                    if str(train_Xp[c].dtype) == "interval"
                ]
                for c in interval_cols:
                    train_Xp[c] = train_Xp[c].astype("string")
                    val_Xp[c]   = val_Xp[c].astype("string")

            if model == "catboost" and not PARAMS["one_hot"]:
                for c in train_Xp.columns:
                    if train_Xp[c].dtype == "object":
                        s = train_Xp[c].astype(str)
                        if s.str.match(r"^\(.*,\s*.*\]$").any():
                            train_Xp[c] = train_Xp[c].astype("string")
                            val_Xp[c]   = val_Xp[c].astype("string")
            
            PARAMS_FOLD = PARAMS
            
            if model.startswith("lightgbm"):
                PARAMS_FOLD = PARAMS.copy()
            
            if model == "catboost" and not PARAMS["one_hot"]:
                for c in train_Xp.columns:
                    if not is_numeric_dtype(train_Xp[c]):
                        train_Xp[c] = train_Xp[c].astype("string")
                        val_Xp[c]   = val_Xp[c].astype("string")
                cb_cat_cols = [
                    c for c in train_Xp.columns
                    if train_Xp[c].dtype.name in ("object", "string", "category")
                ]
                cb_cat_features = cat_feature_indices_after_preprocessing(
                    train_Xp, cb_cat_cols
                )
                ml_model = fit_catboost(
                    train_Xp, train_y, PARAMS, cat_features=cb_cat_features
                )
                ml_model_scores = evaluate_catboost(
                    ml_model, val_Xp, val_y, cat_features=cb_cat_features
                )
            else:
                ml_model = MODEL_REG[model].fit(train_Xp, train_y, PARAMS_FOLD)
                ml_model_scores = MODEL_REG[model].evaluate(
                    ml_model, val_Xp, val_y
                )

            cv_scores_list.append(ml_model_scores)

            if model == "catboost" and not PARAMS["one_hot"]:
                p_val = predict_catboost(
                    ml_model, val_Xp, cat_features=cb_cat_features
                )
            else:
                p_val = MODEL_REG[model].predict(ml_model, val_Xp)

            oof_proba.loc[val_df.index] = p_val

        oof_pred_all[model] = oof_proba.astype(float)

        # Fit full model

        if RU_CONFIG["fit_model"]:

            train_X, train_y = split_dataset(train_data_prepared, label_col)

            pre_pipe = build_preprocess_pipeline(
                num_cols=num_cols,
                cat_cols=cat_cols,
                yj_cols=yj_cols,
                standardize=bool(RU_CONFIG["standardize"]),
                one_hot=PARAMS["one_hot"],
                augment=PARAMS["augment"],
                engineered_cols=engineered_cols,
                interactions=interactions,
                poly_features=poly_features,
                winsor_cols=winsor_features,
                augment_blocks=["squares", "log1p", "ratios"],
                augment_squares_for=[
                    "cholesterol",
                    "blood_pressure",
                    "st_depression",
                    "max_hr"
                ],
                augment_log1p_for=[
                    "cholesterol",
                    "blood_pressure",
                    "st_depression"
                ],
                augment_ratios=[
                    ("cholesterol_over_blood_pressure",
                     "cholesterol",
                     "blood_pressure"),
                    ("st_depression_over_max_hr",
                     "st_depression",
                     "max_hr"),
                    ("max_hr_over_age",
                     "max_hr",
                     "age"),
                ]
            )

            train_Xp = pre_pipe.fit_transform(train_X, train_y)
            test_ids = test_data_converted["id"].to_numpy()
            test_X = test_data_prepared
            test_Xp = pre_pipe.transform(test_X)

            drop_features = build_drop_features(
                list(train_Xp.columns), additional_features_cfg
            )

            train_Xp = train_Xp.drop(
               columns=[c for c in drop_features if c in train_Xp.columns]
            )
            test_Xp = test_Xp.drop(
               columns=[c for c in drop_features if c in test_Xp.columns]
            )
            test_Xp = test_Xp.reindex(columns=train_Xp.columns, fill_value=0.0)
            
            if model == "catboost" and not PARAMS["one_hot"]:
                interval_cols = [
                    c for c in train_Xp.columns
                    if str(train_Xp[c].dtype) == "interval"
                ]
                for c in interval_cols:
                    train_Xp[c] = train_Xp[c].astype("string")
                    test_Xp[c]  = test_Xp[c].astype("string")
            
            if model == "catboost" and not PARAMS["one_hot"]:
                for c in train_Xp.columns:
                    if train_Xp[c].dtype == "object":
                        s = train_Xp[c].astype(str)
                        if s.str.match(r"^\(.*,\s*.*\]$").any():
                            train_Xp[c] = train_Xp[c].astype("string")
                            test_Xp[c]   = test_Xp[c].astype("string")
                
            PARAMS_FULL = PARAMS
            if model.startswith("lightgbm"):
                PARAMS_FULL = PARAMS.copy()
            
            if model == "catboost" and not PARAMS["one_hot"]:
                for c in train_Xp.columns:
                    if not is_numeric_dtype(train_Xp[c]):
                        train_Xp[c] = train_Xp[c].astype("string")
                        test_Xp[c]   = test_Xp[c].astype("string")
                cb_cat_cols = [
                    c for c in train_Xp.columns
                    if train_Xp[c].dtype.name
                    in ("object", "string", "category")
                ]
                cb_cat_features = cat_feature_indices_after_preprocessing(
                    train_Xp, cb_cat_cols
                )

                full_ml_model = fit_catboost(
                    train_Xp, train_y, PARAMS, cat_features=cb_cat_features
                )
                y_proba = predict_catboost(
                    full_ml_model, test_Xp, cat_features=cb_cat_features
                )
            else:
                full_ml_model = MODEL_REG[model].fit(
                    train_Xp, train_y, PARAMS_FULL
                )
                y_proba = MODEL_REG[model].predict(full_ml_model, test_Xp)

            # Store regression table

            if (
                model == "logistic_regression" and
                RU_CONFIG["store_regression"]
            ):
                store_regression_fn = globals().get("store_regression_table")
                if callable(store_regression_fn):
                    store_regression_fn(train_Xp, train_y, output_dir)
                else:
                    print(
                        "Skipping regression table export: "
                        "store_regression_table is unavailable."
                    )

            # Store model

            if store_model:
                MODEL_REG[model].store(full_ml_model, dst_dir, model)

            # Predict values for test data
            
            col_name_1 = f"id_{model}"
            col_name_2 = f"y_proba_{model}"

            output_df[col_name_1] = test_ids
            output_df[col_name_2] = y_proba
    
    # Store outputs

    ##  CV scores

    cv_scores = pd.concat(cv_scores_list, axis=0, ignore_index=True)
    cv_scores_table = cv_scores.describe().round(5)
    print(cv_scores_table)
    cv_scores_table.to_csv(output_dir / "cv_scores.csv", index=False)

    ## Full model scores

    stack = None

    if RU_CONFIG["fit_model"]:
        y_true = train_data_prepared[PR_CONFIG["label_col"]]

        oof_cols = [
            m for m in RU_CONFIG["models"]
            if (m in oof_pred_all.columns)
            and (f"y_proba_{m}" in output_df.columns)
        ]

        X_oof = oof_pred_all[oof_cols]
        mask = X_oof.notna().all(axis=1)
        oof_mean = X_oof.loc[mask].mean(axis=1)
        full_score = evaluate_ensemble(y_true.loc[mask], oof_mean)

        with open(output_dir / "full_score.csv", "w") as f:
            f.write(f"{full_score:.5f}\n")

    ## Predicted probabilities

    base_cols = [
        f"y_proba_{m}" for m in RU_CONFIG["models"]
        if f"y_proba_{m}" in output_df.columns
    ]

    output_df["final_predictions"] = 0.0
    num_models = 0
    for col in base_cols:
        num_models += 1
        output_df["final_predictions"] += output_df[col]
    output_df["final_predictions"] = (
        output_df["final_predictions"] / max(num_models, 1)
    )

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
