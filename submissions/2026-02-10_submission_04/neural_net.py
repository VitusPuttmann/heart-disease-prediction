"""
Functions for the training, evaluation and prediction for a Tensor Flow neural
net.
"""

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


NN_PARAMS: dict[str, object] = {
    "hidden_units": [64, 32],
    "activation": "relu",
    "dropout": 0.0,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 50,
    "verbose": 2,
    "seed": 834217
}


def _build_model(n_features: int, params: dict[str, object]) -> keras.Model:
    tf.keras.utils.set_random_seed(int(params["seed"]))

    inputs = keras.Input(shape=(n_features,))
    x = inputs

    for units in params["hidden_units"]:
        x = layers.Dense(
            int(units), activation=(params["activation"])
        )(x)
    
        drop = float(params["dropout"])
        if drop > 0:
            x = layers.Dropout(drop)(x)
    
    outputs = layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    lr = params["learning_rate"]
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[keras.metrics.MeanSquaredError(name="mse")]
    )

    return model


def fit_neuralnet(
        train_X: pd.DataFrame,
        train_y: pd.Series,
        params: dict[str, object]
    ) -> keras.Model:
    """
    Fit a TensorFlow/Keras neural network regression model.
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
    Evaluate a previously trained neural network model and return MSE.
    """

    X = val_X.to_numpy(dtype=np.float32)
    y = val_y.to_numpy(dtype=np.float32).reshape(-1, 1)

    loss, mse = model.evaluate(X, y, verbose=2)

    scores = pd.DataFrame(
        [[float(mse)]],
        columns=["mse"]
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
    y_pred = model.predict(X, verbose=1).reshape(-1)

    return pd.Series(y_pred)
