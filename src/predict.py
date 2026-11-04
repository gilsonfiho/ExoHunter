"""Gera as classificações do modelo para TODOS os KOIs.

Carrega o modelo treinado e os transformadores (imputer + scaler) salvos no
treino, pontua todos os objetos — inclusive os CANDIDATE, que não têm rótulo —
e salva ``data/processed/predictions.csv`` para o dashboard consumir.

Uso:
    python -m src.predict
"""
from __future__ import annotations

import sys

# IMPORTANTE: importar o TensorFlow ANTES de joblib/numpy/pandas. No Windows,
# essas libs carregam DLLs nativas que entram em conflito com o runtime do TF
# se subirem primeiro ("DLL load failed ... _pywrap_tensorflow_internal").
from tensorflow import keras  # noqa: E402  (import propositalmente no topo)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

PREDICTIONS_CSV = config.DATA_PROCESSED / "predictions.csv"

# Colunas físicas extras que exibimos no dashboard (além das features do modelo).
DISPLAY_EXTRA = ["koi_period", "koi_prad", "koi_teq", "koi_depth", "koi_model_snr"]


def _load_artifacts() -> tuple[keras.Model, object, object]:
    """Carrega modelo, imputer e scaler persistidos no treino."""
    model_path = config.MODELS_DIR / "exoplanet_mlp.keras"
    imputer_path = config.DATA_PROCESSED / "imputer.joblib"
    scaler_path = config.DATA_PROCESSED / "scaler.joblib"

    missing = [p for p in (model_path, imputer_path, scaler_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Artefatos de treino não encontrados: "
            + ", ".join(str(p) for p in missing)
            + "\nRode primeiro: python -m src.train"
        )
    return keras.models.load_model(model_path), joblib.load(imputer_path), joblib.load(scaler_path)


def generate_predictions() -> pd.DataFrame:
    """Pontua todos os KOIs e devolve um DataFrame pronto para exibição."""
    from .data_collection import collect

    df = collect()  # usa o CSV em cache
    model, imputer, scaler = _load_artifacts()

    X = scaler.transform(imputer.transform(df[config.FEATURES].to_numpy(dtype="float64")))
    prob = model.predict(X, verbose=0).ravel()

    out = df[["kepoi_name", "kepler_name", config.TARGET_COL] + DISPLAY_EXTRA].copy()
    out["prob_exoplaneta"] = prob
    out["classificacao_modelo"] = np.where(
        prob >= 0.5, config.POSITIVE_CLASS, config.NEGATIVE_CLASS
    )

    # Para os objetos já rotulados, marca se o modelo acertou.
    rotulado = out[config.TARGET_COL].isin([config.POSITIVE_CLASS, config.NEGATIVE_CLASS])
    out["acertou"] = np.where(
        rotulado, out["classificacao_modelo"] == out[config.TARGET_COL], np.nan
    )

    out.to_csv(PREDICTIONS_CSV, index=False)
    print(f"[predict] {len(out):,} KOIs classificados -> {PREDICTIONS_CSV}")
    return out


def main(argv: list[str] | None = None) -> int:
    df = generate_predictions()
    print("\n[predict] Classificação do modelo (todos os KOIs):")
    print(df["classificacao_modelo"].value_counts().to_string())
    cand = df[df[config.TARGET_COL] == "CANDIDATE"]
    prev_planetas = (cand["classificacao_modelo"] == config.POSITIVE_CLASS).sum()
    print(f"\n[predict] Dos {len(cand):,} candidatos, o modelo prevê "
          f"{prev_planetas:,} como exoplanetas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
