"""
Execution entry point of the ML project.
"""

import time

from pathlib import Path
import shutil
import yaml

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.loading import load_raw_data
from src.preparation import rename_features, numeric_to_string
from src.features import (
    build_preprocess_pipeline
)
from src.cross_validation import iter_cv_folds
from src.training import split_dataset
from src.models.catboost import (
    fit_catboost,
    evaluate_catboost,
    predict_catboost,
    store_catboost
)
from src.models.lightgbm import (
    fit_lightgbm,
    evaluate_lightgbm,
    predict_lightgbm,
    store_lightgbm,
    build_monotone_constraints
)
from src.models.logistic_regression import (
    fit_logistic_regression,
    evaluate_logistic_regression,
    predict_logistic_regression,
    store_logistic_regression
)
from src.models.neural_net import (
    fit_neuralnet,
    evaluate_neuralnet,
    predict_neuralnet,
    store_neural_net
)
from src.models.real_mlp import (
    fit_realmlp,
    evaluate_realmlp,
    predict_realmlp,
    store_realmlp
)
from src.models.xgboost import (
    fit_xgboost,
    evaluate_xgboost,
    predict_xgboost,
    store_xgboost
)
from src.ensemble import evaluate_ensemble


with open("configs/run.yaml", "r") as f:
    RU_CONFIG = yaml.safe_load(f)
with open("configs/project.yaml", "r") as f:
    PR_CONFIG = yaml.safe_load(f)
with open("configs/cross_validation.yaml", "r") as f:
    CV_CONFIG = yaml.safe_load(f)
with open("configs/features.yaml", "r") as f:
    FEATURES = yaml.safe_load(f)
with open("params/catboost.yaml", "r") as f:
    CB_PARAMS = yaml.safe_load(f)
with open("params/lightgbm.yaml", "r") as f:
    LG_PARAMS = yaml.safe_load(f)
with open("params/logistic_regression.yaml", "r") as f:
    LR_PARAMS = yaml.safe_load(f)
with open("params/neural_net.yaml", "r") as f:
    NN_PARAMS = yaml.safe_load(f)
with open("params/real_mlp.yaml", "r") as f:
    RM_PARAMS = yaml.safe_load(f)
with open("params/xgboost.yaml", "r") as f:
    XG_PARAMS = yaml.safe_load(f)

# REFACTOR START: Check whether necessary
FEATURE_KEYS = {
    k for k, v in FEATURES.items() if isinstance(v, dict) and "name_clean" in v
}
# REFACTOR END

# REFACTOR START: Check whether necessary
def cat_feature_indices_after_preprocessing(
        train_Xp: pd.DataFrame, cat_cols: list[str]
    ) -> list[int]:
    return [
        train_Xp.columns.get_loc(c) for c in cat_cols if c in train_Xp.columns
    ] # type: ignore
# REFACTOR END

# REFACTOR START: Check whether necessary
def is_engineered(k: str) -> bool:
    return isinstance(
        FEATURES.get(k), dict
    ) and FEATURES[k].get("source") == "engineered"
# REFACTOR END

CV_CONFIG["n_splits"] = RU_CONFIG["cv_splits"]

# REFACTOR START: Shift to separate file in params folder
MODEL_DICT = {
    "lightgbm_main": {
        "params":       LG_PARAMS["main"],
        "fit":          fit_lightgbm,
        "eval":         evaluate_lightgbm,
        "pred":         predict_lightgbm,
        "store":        store_lightgbm
    },
    "lightgbm_deep_expressive": {
        "params":       LG_PARAMS["deep_expressive"],
        "fit":          fit_lightgbm,
        "eval":         evaluate_lightgbm,
        "pred":         predict_lightgbm,
        "store":        store_lightgbm
    },
    "lightgbm_strong_reg": {
        "params":       LG_PARAMS["strong_reg"],
        "fit":          fit_lightgbm,
        "eval":         evaluate_lightgbm,
        "pred":         predict_lightgbm,
        "store":        store_lightgbm
    },
    "lightgbm_boost": {
        "params":       LG_PARAMS["boost"],
        "fit":          fit_lightgbm,
        "eval":         evaluate_lightgbm,
        "pred":         predict_lightgbm,
        "store":        store_lightgbm
    },
    "lightgbm_more_subs": {
        "params":       LG_PARAMS["more_subs"],
        "fit":          fit_lightgbm,
        "eval":         evaluate_lightgbm,
        "pred":         predict_lightgbm,
        "store":        store_lightgbm
    },
    "catboost": {
        "params":       CB_PARAMS,
        "fit":          fit_catboost,
        "eval":         evaluate_catboost,
        "pred":         predict_catboost,
        "store":        store_catboost
    },
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
# REFACTOR END


# REFACTOR CURRENT POSITION

drop_features = [
    'age_x_number_vessels__age_bin_10y_(35.0, 45.0]__x__number_vessels_two',
    'age_x_number_vessels__age_bin_10y_(35.0, 45.0]__x__number_vessels_zero',
    'age_x_number_vessels__age_bin_10y_(45.0, 55.0]__x__number_vessels_three',
    'age_x_number_vessels__age_bin_10y_(45.0, 55.0]__x__number_vessels_two',
    'age_x_number_vessels__age_bin_10y_(45.0, 55.0]__x__number_vessels_zero',
    'age_x_number_vessels__age_bin_10y_(55.0, 65.0]__x__number_vessels_three',
    'age_x_number_vessels__age_bin_10y_(55.0, 65.0]__x__number_vessels_two',
    'age_x_number_vessels__age_bin_10y_(55.0, 65.0]__x__number_vessels_zero',
    'age_x_number_vessels__age_bin_10y_(65.0, 75.0]__x__number_vessels_three',
    'age_x_number_vessels__age_bin_10y_(65.0, 75.0]__x__number_vessels_two',
    'age_x_number_vessels__age_bin_10y_(65.0, 75.0]__x__number_vessels_zero',
    'age_x_number_vessels__age_bin_10y_(75.0, 85.0]__x__number_vessels_three',
    'age_x_number_vessels__age_bin_10y_(75.0, 85.0]__x__number_vessels_two',
    'age_x_number_vessels__age_bin_10y_(75.0, 85.0]__x__number_vessels_zero',
    'age_x_number_vessels__age_bin_5y_(30.0, 35.0]__x__number_vessels_three',
    'age_x_number_vessels__age_bin_5y_(30.0, 35.0]__x__number_vessels_two',
    'age_x_number_vessels__age_bin_5y_(30.0, 35.0]__x__number_vessels_zero',
    'age_x_number_vessels__age_bin_5y_(35.0, 40.0]__x__number_vessels_three',
    'age_x_number_vessels__age_bin_5y_(35.0, 40.0]__x__number_vessels_two',
    'age_x_number_vessels__age_bin_5y_(35.0, 40.0]__x__number_vessels_zero',
    'age_x_number_vessels__age_bin_5y_(40.0, 45.0]__x__number_vessels_three',
    'age_x_number_vessels__age_bin_5y_(40.0, 45.0]__x__number_vessels_two',
    'age_x_number_vessels__age_bin_5y_(40.0, 45.0]__x__number_vessels_zero',
    'age_x_number_vessels__age_bin_5y_(45.0, 50.0]__x__number_vessels_three',
    'age_x_number_vessels__age_bin_5y_(45.0, 50.0]__x__number_vessels_two',
    'age_x_number_vessels__age_bin_5y_(45.0, 50.0]__x__number_vessels_zero',
    'age_x_number_vessels__age_bin_5y_(50.0, 55.0]__x__number_vessels_three',
    'age_x_number_vessels__age_bin_5y_(50.0, 55.0]__x__number_vessels_two',
    'age_x_number_vessels__age_bin_5y_(50.0, 55.0]__x__number_vessels_zero',
    'age_x_number_vessels__age_bin_5y_(55.0, 60.0]__x__number_vessels_three',
    'age_x_number_vessels__age_bin_5y_(55.0, 60.0]__x__number_vessels_two',
    'age_x_number_vessels__age_bin_5y_(55.0, 60.0]__x__number_vessels_zero',
    'age_x_number_vessels__age_bin_5y_(60.0, 65.0]__x__number_vessels_three',
    'age_x_number_vessels__age_bin_5y_(60.0, 65.0]__x__number_vessels_two',
    'age_x_number_vessels__age_bin_5y_(60.0, 65.0]__x__number_vessels_zero',
    'age_x_number_vessels__age_bin_5y_(65.0, 70.0]__x__number_vessels_three',
    'age_x_number_vessels__age_bin_5y_(65.0, 70.0]__x__number_vessels_two',
    'age_x_number_vessels__age_bin_5y_(65.0, 70.0]__x__number_vessels_zero',
    'age_x_number_vessels__age_bin_5y_(70.0, 75.0]__x__number_vessels_three',
    'age_x_number_vessels__age_bin_5y_(70.0, 75.0]__x__number_vessels_two',
    'age_x_number_vessels__age_bin_5y_(70.0, 75.0]__x__number_vessels_zero',
    'age_x_number_vessels__age_bin_5y_(75.0, 80.0]__x__number_vessels_three',
    'age_x_number_vessels__age_bin_5y_(75.0, 80.0]__x__number_vessels_two',
    'age_x_number_vessels__age_bin_5y_(75.0, 80.0]__x__number_vessels_zero',
    'age_x_thallium__age_bin_10y_(35.0, 45.0]__x__thallium_normal',
    'age_x_thallium__age_bin_10y_(35.0, 45.0]__x__thallium_reversible defect',
    'age_x_thallium__age_bin_10y_(45.0, 55.0]__x__thallium_normal',
    'age_x_thallium__age_bin_10y_(45.0, 55.0]__x__thallium_reversible defect',
    'age_x_thallium__age_bin_10y_(55.0, 65.0]__x__thallium_normal',
    'age_x_thallium__age_bin_10y_(55.0, 65.0]__x__thallium_reversible defect',
    'age_x_thallium__age_bin_10y_(65.0, 75.0]__x__thallium_normal',
    'age_x_thallium__age_bin_10y_(65.0, 75.0]__x__thallium_reversible defect',
    'age_x_thallium__age_bin_10y_(75.0, 85.0]__x__thallium_normal',
    'age_x_thallium__age_bin_10y_(75.0, 85.0]__x__thallium_reversible defect',
    'age_x_thallium__age_bin_5y_(30.0, 35.0]__x__thallium_normal',
    'age_x_thallium__age_bin_5y_(30.0, 35.0]__x__thallium_reversible defect',
    'age_x_thallium__age_bin_5y_(35.0, 40.0]__x__thallium_normal',
    'age_x_thallium__age_bin_5y_(35.0, 40.0]__x__thallium_reversible defect',
    'age_x_thallium__age_bin_5y_(40.0, 45.0]__x__thallium_normal',
    'age_x_thallium__age_bin_5y_(40.0, 45.0]__x__thallium_reversible defect',
    'age_x_thallium__age_bin_5y_(45.0, 50.0]__x__thallium_normal',
    'age_x_thallium__age_bin_5y_(45.0, 50.0]__x__thallium_reversible defect',
    'age_x_thallium__age_bin_5y_(50.0, 55.0]__x__thallium_normal',
    'age_x_thallium__age_bin_5y_(50.0, 55.0]__x__thallium_reversible defect',
    'age_x_thallium__age_bin_5y_(55.0, 60.0]__x__thallium_normal',
    'age_x_thallium__age_bin_5y_(55.0, 60.0]__x__thallium_reversible defect',
    'age_x_thallium__age_bin_5y_(60.0, 65.0]__x__thallium_normal',
    'age_x_thallium__age_bin_5y_(60.0, 65.0]__x__thallium_reversible defect',
    'age_x_thallium__age_bin_5y_(65.0, 70.0]__x__thallium_normal',
    'age_x_thallium__age_bin_5y_(65.0, 70.0]__x__thallium_reversible defect',
    'age_x_thallium__age_bin_5y_(70.0, 75.0]__x__thallium_normal',
    'age_x_thallium__age_bin_5y_(70.0, 75.0]__x__thallium_reversible defect',
    'age_x_thallium__age_bin_5y_(75.0, 80.0]__x__thallium_normal',
    'age_x_thallium__age_bin_5y_(75.0, 80.0]__x__thallium_reversible defect',
    'age_x_exercise_angina__age_bin_10y_(35.0, 45.0]__x__exercise_angina_True',
    'age_x_exercise_angina__age_bin_10y_(45.0, 55.0]__x__exercise_angina_True',
    'age_x_exercise_angina__age_bin_10y_(55.0, 65.0]__x__exercise_angina_True',
    'age_x_exercise_angina__age_bin_10y_(65.0, 75.0]__x__exercise_angina_True',
    'age_x_exercise_angina__age_bin_10y_(75.0, 85.0]__x__exercise_angina_True',
    'age_x_exercise_angina__age_bin_5y_(30.0, 35.0]__x__exercise_angina_True',
    'age_x_exercise_angina__age_bin_5y_(35.0, 40.0]__x__exercise_angina_True',
    'age_x_exercise_angina__age_bin_5y_(40.0, 45.0]__x__exercise_angina_True',
    'age_x_exercise_angina__age_bin_5y_(45.0, 50.0]__x__exercise_angina_True',
    'age_x_exercise_angina__age_bin_5y_(50.0, 55.0]__x__exercise_angina_True',
    'age_x_exercise_angina__age_bin_5y_(55.0, 60.0]__x__exercise_angina_True',
    'age_x_exercise_angina__age_bin_5y_(60.0, 65.0]__x__exercise_angina_True',
    'age_x_exercise_angina__age_bin_5y_(65.0, 70.0]__x__exercise_angina_True',
    'age_x_exercise_angina__age_bin_5y_(70.0, 75.0]__x__exercise_angina_True',
    'age_x_exercise_angina__age_bin_5y_(75.0, 80.0]__x__exercise_angina_True',
    'age_x_number_vessels__age_bin_10y_(35.0, 45.0]__x__number_vessels_three',
    'num_rest__mean_bp_by_exercise_angina',
    'num_rest__mean_chol_by_exercise_angina',
    'num_rest__mean_hr_by_exercise_angina',
    'num_rest__mean_stdep_by_exercise_angina',
    'num_rest__mean_age_by_exercise_angina',
    'num_rest__mean_bp_by_thallium',
    'num_rest__mean_chol_by_thallium',
    'num_rest__mean_hr_by_thallium',
    'num_rest__mean_stdep_by_thallium',
    'num_rest__mean_age_by_thallium',
    'num_rest__mean_bp_by_number_vessels',
    'num_rest__mean_chol_by_number_vessels',
    'num_rest__mean_hr_by_number_vessels',
    'num_rest__mean_stdep_by_number_vessels',
    'num_rest__mean_age_by_number_vessels',
    'num_rest__mean_bp_by_sex',
    'num_rest__mean_chol_by_sex',
    'num_rest__mean_hr_by_sex',
    'num_rest__mean_stdep_by_sex',
    'num_rest__mean_age_by_sex',
    'num_rest__mean_bp_by_chest_pain',
    'num_rest__mean_chol_by_chest_pain',
    'num_rest__mean_hr_by_chest_pain',
    'num_rest__mean_stdep_by_chest_pain',
    'num_rest__mean_age_by_chest_pain',
    'num_rest__mean_bp_by_ekg',
    'num_rest__mean_chol_by_ekg',
    'num_rest__mean_hr_by_ekg',
    'num_rest__mean_stdep_by_ekg',
    'num_rest__mean_age_by_ekg',
    'num_rest__mean_bp_by_slope_st',
    'num_rest__mean_chol_by_slope_st',
    'num_rest__mean_hr_by_slope_st',
    'num_rest__mean_stdep_by_slope_st',
    'num_rest__mean_age_by_slope_st',
    'num_rest__cholesterol_over_blood_pressure',
    'num_rest__blood_pressure_over_max_hr',
    'num_rest__cholesterol_over_max_hr',
    'num_rest__st_depression_over_max_hr',
    'num_rest__max_hr_over_age',
    'num_rest__blood_pressure_over_age',
    'num_rest__cholesterol_over_age',
    'num_rest__st_depression_over_age',
    'num_rest__age_x_max_hr',
    'num_rest__age_x_st_depression',
    'num_rest__max_hr_x_st_depression',
    'num_rest__blood_pressure_x_st_depression',
    'num_rest__cholesterol_x_st_depression',
    'num_rest__blood_pressure_x_max_hr',
    'num_rest__cholesterol_x_max_hr',
    'num_rest__cnt_by_sex',
    'num_rest__cnt_by_chest_pain',
    'num_rest__cnt_by_exercise_angina',
    'num_rest__cnt_by_thallium',
    'num_rest__cnt_by_number_vessels',
    'num_rest__cnt_by_ekg',
    'num_rest__cnt_by_slope_st',
    'num_rest__cnt_by_age_bin_10y_sex',
    'num_rest__cnt_by_thallium_number_vessels',
    'num_rest__cnt_by_sex_chest_pain',
    'num_rest__cnt_by_exercise_angina_slope_st',
]

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

    MONO_SPEC = {
        "number_vessels": +1,
        "st_depression": +1,
        "exercise_angina": +1,
        "thallium_fixed_defect": +1,
        "thallium_reversible_defect": +1,
    }
    # Select features

    train_data_prepared = train_data_converted[
        selected_feature_columns + [label_col]
    ]
    test_data_prepared  = test_data_converted[selected_feature_columns]

    # Train model

    ## Initiate output

    cv_scores_list = []

    cv_scores_filt_list = []

    oof_pred_all = pd.DataFrame(index=train_data_prepared.index)

    output_df = pd.DataFrame()

    ## Create model loop

    for model in RU_CONFIG["models"]:
        
        ## Obtain parameters

        PARAMS = MODEL_DICT[model]["params"]

        ## Create CV iterator
        
        cv_iterator = iter_cv_folds(train_data_prepared, label_col, CV_CONFIG)

        ## Train and evaluate model with CV

        oof_proba = pd.Series(index=train_data_prepared.index, dtype=float)

        for fold_idx, train_df, val_df in cv_iterator:
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
                augment_squares_for=["cholesterol", "blood_pressure", "st_depression", "max_hr"],
                augment_log1p_for=["cholesterol", "blood_pressure", "st_depression"],
                augment_ratios=[
                    ("cholesterol_over_blood_pressure", "cholesterol", "blood_pressure"),
                    ("st_depression_over_max_hr", "st_depression", "max_hr"),
                    ("max_hr_over_age", "max_hr", "age"),
                ]
            )
            
            train_Xp = pre_pipe.fit_transform(train_X, train_y)
            print("Initial number of features:")
            print(len(train_Xp.columns))          
            val_Xp   = pre_pipe.transform(val_X)

            train_Xp = train_Xp.drop(
               columns=[c for c in drop_features if c in train_Xp.columns]
            )
            val_Xp = val_Xp.reindex(columns=train_Xp.columns, fill_value=0.0)
            print("Adapted number of features:")
            print(len(train_Xp.columns))

            if model == "catboost" and not PARAMS["one_hot"]:
                interval_cols = [c for c in train_Xp.columns if str(train_Xp[c].dtype) == "interval"]
                for c in interval_cols:
                    train_Xp[c] = train_Xp[c].astype("string")
                    val_Xp[c]   = val_Xp[c].astype("string")

            if model == "catboost" and not PARAMS["one_hot"]:
                for c in train_Xp.columns:
                    if train_Xp[c].dtype == "object":
                        s = train_Xp[c].astype(str)
                        if s.str.match(r"^\(.*,\s*.*\]$").any():
                            train_Xp[c] = train_Xp[c].astype("string")
                            val_Xp[c]   = val_Xp[c].astype("string")   # CV
            
            PARAMS_FOLD = PARAMS
            
            if model.startswith("lightgbm"):
                PARAMS_FOLD = PARAMS.copy()
                PARAMS_FOLD["monotone_constraints"] = build_monotone_constraints(train_Xp, MONO_SPEC)
                PARAMS_FOLD["monotone_constraints_method"] = "advanced"
            
            if model == "catboost" and not PARAMS["one_hot"]:
                for c in train_Xp.columns:
                    if not is_numeric_dtype(train_Xp[c]):
                        train_Xp[c] = train_Xp[c].astype("string")
                        val_Xp[c]   = val_Xp[c].astype("string")
                cb_cat_cols = [c for c in train_Xp.columns if train_Xp[c].dtype.name in ("object", "string", "category")]
                cb_cat_features = [train_Xp.columns.get_loc(c) for c in cb_cat_cols]
                ml_model = fit_catboost(train_Xp, train_y, PARAMS, cat_features=cb_cat_features)
                ml_model_scores = evaluate_catboost(ml_model, val_Xp, val_y, cat_features=cb_cat_features)
            else:
                ml_model = MODEL_DICT[model]["fit"](train_Xp, train_y, PARAMS_FOLD)
                ml_model_scores = MODEL_DICT[model]["eval"](ml_model, val_Xp, val_y)

            cv_scores_list.append(ml_model_scores)

            if model == "catboost" and not PARAMS["one_hot"]:
                p_val = predict_catboost(ml_model, val_Xp, cat_features=cb_cat_features) # type: ignore
            else:
                p_val = MODEL_DICT[model]["pred"](ml_model, val_Xp)

            oof_proba.loc[val_df.index] = p_val

        oof_pred_all[model] = oof_proba.astype(float)

        # Filter out label disagreemtns

        y = train_data_prepared[label_col]

        t = 0.95
        flag = ((y == 0) & (oof_proba >= t)) | ((y == 1) & (oof_proba <= 1 - t))

        suspicion = pd.Series(0.0, index=y.index)
        
        suspicion.loc[(y == 0)] = oof_proba.loc[(y == 0)]
        suspicion.loc[(y == 1)] = (1 - oof_proba.loc[(y == 1)])

        suspicion = suspicion.where(flag, 0.0)

        q = 0.02  # 2%
        n_drop = int(len(train_data_prepared) * q)

        cand = suspicion[suspicion > 0].sort_values(ascending=False)
        drop_idx = cand.head(min(n_drop, len(cand))).index
        train_data_clean = train_data_prepared.drop(index=drop_idx)

        """
        ## Second run with cleaner data
            
        cv_iterator = iter_cv_folds(train_data_clean, label_col, CV_CONFIG)

        for fold_idx, train_df, val_df in cv_iterator:
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
                augment_squares_for=["cholesterol", "blood_pressure", "st_depression", "max_hr"],
                augment_log1p_for=["cholesterol", "blood_pressure", "st_depression"],
                augment_ratios=[
                    ("cholesterol_over_blood_pressure", "cholesterol", "blood_pressure"),
                    ("st_depression_over_max_hr", "st_depression", "max_hr"),
                    ("max_hr_over_age", "max_hr", "age"),
                ]
            )
                    
            train_Xp = pre_pipe.fit_transform(train_X, train_y)
            val_Xp   = pre_pipe.transform(val_X)

            PARAMS_FOLD = PARAMS
            if model.startswith("lightgbm"):
                PARAMS_FOLD = PARAMS.copy()
                PARAMS_FOLD["monotone_constraints"] = build_monotone_constraints(train_Xp, MONO_SPEC)
                PARAMS_FOLD["monotone_constraints_method"] = "advanced"

            if model == "catboost" and not PARAMS["one_hot"]:
                cb_cat_features = cat_feature_indices_after_preprocessing(train_Xp, cat_cols) # type: ignore
                ml_model = fit_catboost(train_Xp, train_y, PARAMS, cat_features=cb_cat_features)
                ml_model_scores = evaluate_catboost(ml_model, val_Xp, val_y, cat_features=cb_cat_features)
            else:
                ml_model = MODEL_DICT[model]["fit"](train_Xp, train_y, PARAMS_FOLD)
                ml_model_scores = MODEL_DICT[model]["eval"](ml_model, val_Xp, val_y)

            cv_scores_filt_list.append(ml_model_scores)

        """

        # Fit full model
        if RU_CONFIG["fit_model"]:
        
            label_col=PR_CONFIG["label_col"]
            
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
                augment_squares_for=["cholesterol", "blood_pressure", "st_depression", "max_hr"],
                augment_log1p_for=["cholesterol", "blood_pressure", "st_depression"],
                augment_ratios=[
                    ("cholesterol_over_blood_pressure", "cholesterol", "blood_pressure"),
                    ("st_depression_over_max_hr", "st_depression", "max_hr"),
                    ("max_hr_over_age", "max_hr", "age"),
                ]
            )

            train_Xp = pre_pipe.fit_transform(train_X, train_y)
            test_ids = test_data_converted["id"].to_numpy()
            test_X = test_data_prepared
            test_Xp = pre_pipe.transform(test_X)

            train_Xp = train_Xp.drop(columns=[c for c in drop_features if c in train_Xp.columns])
            test_Xp  = test_Xp.reindex(columns=train_Xp.columns, fill_value=0.0)
            
            if model == "catboost" and not PARAMS["one_hot"]:
                interval_cols = [c for c in train_Xp.columns if str(train_Xp[c].dtype) == "interval"]
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
                PARAMS_FULL["monotone_constraints"] = build_monotone_constraints(train_Xp, MONO_SPEC)
                PARAMS_FULL["monotone_constraints_method"] = "advanced"
            
            if model == "catboost" and not PARAMS["one_hot"]:
                for c in train_Xp.columns:
                    if not is_numeric_dtype(train_Xp[c]):
                        train_Xp[c] = train_Xp[c].astype("string")
                        test_Xp[c]   = test_Xp[c].astype("string")
                cb_cat_cols = [c for c in train_Xp.columns if train_Xp[c].dtype.name in ("object", "string", "category")]
                cb_cat_features = [train_Xp.columns.get_loc(c) for c in cb_cat_cols]

                full_ml_model = fit_catboost(train_Xp, train_y, PARAMS, cat_features=cb_cat_features)
                y_proba = predict_catboost(full_ml_model, test_Xp, cat_features=cb_cat_features)
            else:
                full_ml_model = MODEL_DICT[model]["fit"](train_Xp, train_y, PARAMS_FULL)
                y_proba = MODEL_DICT[model]["pred"](full_ml_model, test_Xp)

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

    
    """
    cv_scores_filt = pd.concat(cv_scores_filt_list, axis=0, ignore_index=True)
    cv_scores_filt_table = cv_scores_filt.describe().round(5)
    cv_scores_filt_table.to_csv(output_dir / "cv_scores_filt.csv", index=False)
    """

    ## Full model scores

    stack = None

    if RU_CONFIG["fit_model"]:
        y_true = train_data_prepared[PR_CONFIG["label_col"]]

        oof_cols = [
            m for m in RU_CONFIG["models"]
            if (m in oof_pred_all.columns) and (f"y_proba_{m}" in output_df.columns)
        ]

        ens_cfg = RU_CONFIG.get("ensemble", {"method": "mean"})
        if ens_cfg.get("method") == "stacking_regression":
            X_oof = oof_pred_all[oof_cols].copy()
            y = train_data_prepared[PR_CONFIG["label_col"]].copy()

            mask = X_oof.notna().all(axis=1)
            X_oof = X_oof.loc[mask]
            y = y.loc[mask]

            print("y mean:", float(y.mean()))
            print("OOF base std:", X_oof.std().to_dict())
            for c in X_oof.columns:
                print(c, "min/max", float(X_oof[c].min()), float(X_oof[c].max()))

            stack = fit_stacking_regression(
                oof_pred=X_oof,
                y=y,
                kind=ens_cfg.get("meta_model", "ridge"),
                alpha=float(ens_cfg.get("meta_alpha", 1.0)),
                clip_proba=float(ens_cfg.get("clip_proba", 1e-6)),
                standardize=True,
                use_logit_features=True,   # critical: use logits for both ridge/logistic
            )

            oof_stacked = predict_stacking_regression(stack, X_oof)
            print("OOF stacked:", float(oof_stacked.min()), float(oof_stacked.max()), float(oof_stacked.std()))
            full_score = evaluate_ensemble(y, oof_stacked)

        else:
            X_oof = oof_pred_all[oof_cols]
            mask = X_oof.notna().all(axis=1)
            oof_mean = X_oof.loc[mask].mean(axis=1)
            full_score = evaluate_ensemble(y_true.loc[mask], oof_mean)

        with open(output_dir / "full_score.csv", "w") as f:
            f.write(f"{full_score:.5f}\n")


    ## Predicted probabilities (reuse fitted stack)
    ens_cfg = RU_CONFIG.get("ensemble", {"method": "mean"})
    method = ens_cfg.get("method", "mean")

    base_cols = [
        f"y_proba_{m}" for m in RU_CONFIG["models"]
        if f"y_proba_{m}" in output_df.columns
    ]

    if method == "stacking_regression" and RU_CONFIG["fit_model"]:
        if stack is None:
            raise RuntimeError("Expected fitted stacking model, but stack is None.")

        oof_cols = list(stack.feature_names)  # enforce same order as training

        X_test = pd.DataFrame(
            {m: output_df[f"y_proba_{m}"].astype(float).to_numpy() for m in oof_cols},
            index=output_df.index,
        )

        output_df["final_predictions"] = predict_stacking_regression(stack, X_test).to_numpy()

    else:
        # Fallback: simple mean
        output_df["final_predictions"] = 0.0
        num_models = 0
        for col in base_cols:
            num_models += 1
            output_df["final_predictions"] += output_df[col]
        output_df["final_predictions"] = output_df["final_predictions"] / max(num_models, 1)

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
