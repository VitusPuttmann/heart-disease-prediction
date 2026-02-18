"""
Functions for the training, evaluation and prediction for a neural net.
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

    for units in params["hidden_units"]:
        x = layers.Dense(
            int(units), activation=(params["activation"])
        )(x)
    
        drop = float(params["dropout"])
        if drop > 0:
            x = layers.Dropout(drop)(x)
    
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    lr = params["learning_rate"]
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")]
    )

    return model


def fit_neuralnet(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    params: dict
) -> keras.Model:
    """
    Fit a TensorFlow/Keras neural network binary classification model.
    """

    X = train_X.to_numpy(dtype=np.float32)
    y = train_y.to_numpy(dtype=np.float32).reshape(-1, 1)

    model = _build_model(n_features=X.shape[1], params=params)

    model.fit(
        X, y,
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        verbose=params["verbose"]
    )

    return model


def evaluate_neuralnet(
    model: keras.Model,
    val_X: pd.DataFrame,
    val_y: pd.Series
) -> pd.DataFrame:
    """
    Evaluate a previously trained neural network model and return AUC.
    """

    X = val_X.to_numpy(dtype=np.float32)
    y = val_y.to_numpy(dtype=np.float32).reshape(-1, 1)

    loss, auc = model.evaluate(X, y, verbose=2)  # type: ignore

    scores = pd.DataFrame(
        [[float(auc)]],
        columns=["AUC"]
    )

    return scores


def predict_neuralnet(
    model: keras.Model,
    test_X: pd.DataFrame
) -> pd.Series:
    """
    Predict values for a test dataset based on a previously trained neural net
    model.
    """

    X = test_X.to_numpy(dtype=np.float32)
    y_pred = model.predict(X, verbose=1).reshape(-1) # type: ignore

    return pd.Series(y_pred)


def store_neural_net(
    model: keras.Model,
    filepath: Path,
    model_name: str
) -> None:
    """
    Store a neural net model.
    """

    path = filepath / f"{model_name}.keras"

    model.save(path)
