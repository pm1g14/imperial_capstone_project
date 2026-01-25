import logging
from re import M
from typing import Tuple
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import gen_batch_initial_conditions, optimize_acqf
import pandas as pd
import numpy as np
import tensorflow as tf
import keras.layers as layers
from keras import Sequential
import torch
from botorch.models.model import Model
class NNSurrogateModel:

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


    def _get_tensors_from_dataframe(self) -> Tuple[tf.Tensor, tf.Tensor]:
        df = self._dataset.copy()
        Xs = df.drop('y', axis=1).to_numpy()
        X_tensors = tf.convert_to_tensor(Xs, dtype=tf.float32)
        Y_tensors = tf.convert_to_tensor(df['y'].to_numpy().reshape(-1, 1), dtype=tf.float32)
        return X_tensors, Y_tensors


    def fit_and_predict(self, learning_rate: float, epochs: int) -> Tuple[float, float]:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self._model.compile(loss='mse', optimizer=optimizer)
        self._model.fit(self._X_train, self._Y_train, epochs=epochs, verbose=0)


        return self._model.predict(self._X_train, verbose=0).squeeze(-1)    
        

    def _laplace_nll(self, y_true, y_pred):
        mu = y_pred[:, 0]
        log_b = y_pred[:, 1]
        b = tf.exp(log_b)
        abs_error_scaled = tf.abs(y_true - mu) / b
        nll_per_sample = abs_error_scaled + log_b
        return tf.reduce_mean(nll_per_sample)


    def _gaussian_nll(self, y_true, y_pred):
        # y_pred: [..., 2] -> [mu, log_var]
        mu      = y_pred[:, 0]
        log_var = y_pred[:, 1]
        # If your y_true is shape (batch,) make sure to squeeze/index accordingly
        return tf.reduce_mean(0.5 * tf.exp(-log_var) * tf.square(y_true[:, 0] - mu) + 0.5 * log_var)



    def compute_input_gradients(self, X_new: np.ndarray):
        X_tensor = tf.convert_to_tensor(X_new, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            y_pred = self._model(X_tensor, training=False) 

        grads = tape.batch_jacobian(y_pred, X_tensor)
        grads = tf.squeeze(grads, axis=1)
        return grads.numpy()


    def summarize_gradients(self, X_new: np.ndarray):
        grads = self.compute_input_gradients(X_new)
        feat_influence = np.mean(np.abs(grads), axis=0)
        rank = np.argsort(feat_influence.flatten())[::-1] + 1
        point_steepness = np.linalg.norm(grads, axis=1)
        return feat_influence, rank, point_steepness

    def analyse_point(self):
        x_best = self._X_train[tf.argmin(self._Y_train)] 
        grads = self.compute_input_gradients(x_best[None, :])
        print("Gradients at x_best:", grads.numpy())
