"""
Function for model definition, training and evaluation.
"""

import pandas as pd

import xgboost as xgb
from xgboost import Booster
from sklearn.metrics import accuracy_score


def fit_and_evaluate_decision_tree(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None,
        label_col: str,
        evaluate: bool = True
    ) -> tuple[Booster, pd.Index] | tuple[Booster, float]:
    
    train_X = train_df.drop(columns=[label_col])
    train_y = train_df[label_col]
    
    train_X = pd.get_dummies(train_X, dummy_na=True)
    
    params = {
        "objective": "binary:logistic",
        "max_depth": 3,
        "eta": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 1.0
    }

    if evaluate:
        val_X = val_df.drop(columns=[label_col])
        val_y = val_df[label_col]

        val_X   = pd.get_dummies(val_X, dummy_na=True)

        dtrain = xgb.DMatrix(train_X, label=train_y)
        dval   = xgb.DMatrix(val_X, label=val_y)

        bst = xgb.train(
            {**params, "eval_metric": "error"},
            dtrain,
            num_boost_round=1,
            evals=[(dval, "val")]
        )

        y_pred = (bst.predict(dval) >= 0.5).astype(int)
        accuracy = float(accuracy_score(val_y, y_pred))
        return bst, accuracy

    dtrain = xgb.DMatrix(train_X, label=train_y)
    bst = xgb.train(params, dtrain, num_boost_round=1)
    train_columns = train_X.columns
    return bst, train_columns
