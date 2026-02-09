"""
Functions for the training, evaluation and prediction for a XGBoost tree.
"""

import pandas as pd

import xgboost as xgb
from xgboost import Booster


DC_PARAMS: dict[str, object] = {
    "objective": "binary:logistic",
    "max_depth": 3,
    "eta": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "seed": 726451
}

NUM_BOOST_ROUND = 10


def fit_xgboost_gbdt(
        train_X: pd.DataFrame,
        train_y: pd.Series,
        params: dict[str, object],
    ) -> Booster:
    """
    Fit a XGBoost tree model with the given parameters.
    """

    dtrain = xgb.DMatrix(train_X, label=train_y)
    
    bst = xgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND)
    
    return bst


def evaluate_xgboost_gbdt(
        bst: Booster,
        val_X: pd.DataFrame,
        val_y: pd.Series
    ) -> pd.DataFrame:
    """
    Evaluate a previously trained XGBoost tree model and return error rate.
    """

    dval = xgb.DMatrix(val_X, label=val_y)
    
    y_pred = (bst.predict(dval) > 0.5).astype(int)

    error = float((y_pred != val_y.to_numpy()).mean())
    
    scores = pd.DataFrame(
        [[error]],
        columns=["error"]
    )

    return scores


def predict_xgboost_gbdt(
        bst: Booster,
        test_X: pd.DataFrame,
    ) -> pd.Series:
    """
    Predict values for a test dataset based on a previously trained XGBoost
    tree model.
    """

    dX = xgb.DMatrix(test_X)

    proba = bst.predict(dX)
    y_pred = pd.Series((proba >= 0.5).astype(int))

    return y_pred
