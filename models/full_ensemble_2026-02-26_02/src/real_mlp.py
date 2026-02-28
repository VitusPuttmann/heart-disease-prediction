"""
Functions for the training, evaluation and prediction for a RealMLP model.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import keras
from keras import layers


def _build_model(n_features: int, params: dict) -> keras.Model:
    keras.utils.set_random_seed(int(params["seed"]))

    inputs = keras.Input(shape=(n_features,))
    x = inputs

    activation = params["activation"]
    drop = float(params["dropout"])
    use_residual = bool(params["residual"])

    for units in params["hidden_units"]:
        units = int(units)

        h = layers.Dense(units)(x)
        h = layers.LayerNormalization()(h)
        h = layers.Activation(activation)(h)

        if drop > 0:
            h = layers.Dropout(drop)(h)

        if use_residual and x.shape[-1] == units: # type: ignore
            x = layers.Add()([x, h])
        else:
            x = h

    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    lr = float(params["learning_rate"])
    wd = float(params.get("weight_decay", 0.0))

    try:
        optimizer = keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    except Exception:
        optimizer = keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")]
    )

    return model


def fit_realmlp(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    params: dict
) -> keras.Model:
    """
    Fit a RealMLP-style binary classification model.
    """

    X = train_X.to_numpy(dtype=np.float32)
    y = train_y.to_numpy(dtype=np.float32).reshape(-1, 1)

    model = _build_model(n_features=X.shape[1], params=params)

    model.fit(
        X, y,
        batch_size=int(params["batch_size"]),
        epochs=int(params["epochs"]),
        verbose=int(params["verbose"]) #type: ignore
    )

    return model


def evaluate_realmlp(
    model: keras.Model,
    val_X: pd.DataFrame,
    val_y: pd.Series
) -> pd.DataFrame:
    """
    Evaluate a previously trained RealMLP model and return AUC.
    """

    X = val_X.to_numpy(dtype=np.float32)
    y = val_y.to_numpy(dtype=np.float32).reshape(-1, 1)

    loss, auc = model.evaluate(X, y, verbose=2)  # type: ignore

    return pd.DataFrame([[float(auc)]], columns=["AUC"])


def predict_realmlp(
    model: keras.Model,
    test_X: pd.DataFrame
) -> pd.Series:
    """
    Predict probabilities for a test dataset based on a previously trained 
    RealMLP model.
    """

    X = test_X.to_numpy(dtype=np.float32)
    y_pred = model.predict(X, verbose=1).reshape(-1)  # type: ignore

    return pd.Series(y_pred)


def store_realmlp(
    model: keras.Model,
    filepath: Path,
    model_name: str
) -> None:
    """
    Store a RealMLP model.
    """

    path = filepath / f"{model_name}.keras"
    model.save(path)
