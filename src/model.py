"""Arquitetura da rede neural (Keras/TensorFlow) para classificar KOIs.

Uma MLP (perceptron multicamadas) com BatchNormalization e Dropout para
regularização. Problema binário -> saída sigmoide.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import config


def build_model(input_dim: int, learning_rate: float = 1e-3) -> keras.Model:
    """Cria e compila a MLP.

    Args:
        input_dim: número de features de entrada.
        learning_rate: taxa de aprendizado do otimizador Adam.
    """
    tf.random.set_seed(config.RANDOM_STATE)

    model = keras.Sequential(
        [
            keras.Input(shape=(input_dim,), name="features"),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid", name="prob_exoplaneta"),
        ],
        name="exoplanet_mlp",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model
