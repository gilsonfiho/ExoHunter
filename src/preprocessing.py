"""Pré-processamento dos KOIs para o modelo de deep learning.

Fluxo:
    1. Filtra para o problema binário (CONFIRMED vs FALSE POSITIVE).
    2. Seleciona as features físicas e imputa valores ausentes (mediana).
    3. Faz split estratificado treino/validação/teste.
    4. Padroniza as features (StandardScaler, ajustado só no treino).

Retorna os arrays prontos e persiste o scaler para reuso em inferência.
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config


@dataclass
class Dataset:
    """Conjuntos prontos para treino/avaliação."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


def load_binary_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém apenas as duas classes rotuladas e cria a coluna alvo 0/1."""
    mask = df[config.TARGET_COL].isin([config.POSITIVE_CLASS, config.NEGATIVE_CLASS])
    df = df.loc[mask].copy()
    df["label"] = (df[config.TARGET_COL] == config.POSITIVE_CLASS).astype(int)
    return df


def prepare(df: pd.DataFrame, persist: bool = True) -> Dataset:
    """Transforma o DataFrame bruto nos arrays finais de treino/val/teste."""
    df = load_binary_frame(df)

    X = df[config.FEATURES].to_numpy(dtype="float64")
    y = df["label"].to_numpy(dtype="int64")

    # Split estratificado: primeiro separa teste, depois validação do treino.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        stratify=y, random_state=config.RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=config.VAL_SIZE,
        stratify=y_trainval, random_state=config.RANDOM_STATE,
    )

    # Imputação (mediana) e padronização ajustadas SÓ no treino, para não
    # vazar informação do teste.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = scaler.fit_transform(imputer.fit_transform(X_train))
    X_val = scaler.transform(imputer.transform(X_val))
    X_test = scaler.transform(imputer.transform(X_test))

    if persist:
        joblib.dump(imputer, config.DATA_PROCESSED / "imputer.joblib")
        joblib.dump(scaler, config.DATA_PROCESSED / "scaler.joblib")

    print(f"[prep] treino={len(y_train)}  val={len(y_val)}  teste={len(y_test)}")
    print(f"[prep] positivos (confirmados) no treino: {y_train.mean():.1%}")

    return Dataset(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        feature_names=list(config.FEATURES),
    )
