import pandas as pd
import logging
from re import M
from typing import Tuple
import pandas as pd
import numpy as np
import tensorflow as tf
import keras.layers as layers
from keras import Sequential
import torch

class NNClassifierModel:

    def __init__(self, dataset: pd.DataFrame, hidden_activation: str, output_activation: str):
        self._dataset = dataset
        self._X_train, self._Y_train = self._get_tensors_from_dataframe()
        n_features = self._X_train.shape[1]
        self._model = Sequential([
            layers.Input(shape=(n_features,)),
            layers.Dense(units=16, activation=hidden_activation, use_bias=True),
            layers.Dense(units=8, activation=hidden_activation, use_bias=True),
            layers.Dense(units=1, activation=output_activation, use_bias=True)
        ])


    def fit_and_predict(self, learning_rate: float, epochs: int) -> Tuple[float, float]:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self._model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self._model.fit(self._X_train, self._Y_train, epochs=epochs, verbose=0)
        return self._model.predict(self._X_train, verbose=0).squeeze(-1)


    def _get_tensors_from_dataframe(self) -> Tuple[tf.Tensor, tf.Tensor]:
        df = self._dataset.copy()
        Xs = df.drop('y', axis=1).to_numpy()
        X_tensors = tf.convert_to_tensor(Xs, dtype=tf.float32)
        Y_tensors = tf.convert_to_tensor(df['y'].to_numpy().reshape(-1, 1), dtype=tf.float32)
        return X_tensors, Y_tensors