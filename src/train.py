"""Treina e avalia a rede neural de detecção de exoplanetas.

Pipeline completo:
    coleta (cache) -> pré-processamento -> treino -> avaliação -> artefatos.

Uso:
    python -m src.train                 # treino padrão
    python -m src.train --epochs 60      # ajustar nº de épocas
"""
from __future__ import annotations

import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")  # backend sem display, para salvar figuras em arquivo
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from . import config
from .data_collection import collect
from .model import build_model
from .preprocessing import prepare


def plot_history(history: keras.callbacks.History) -> None:
    """Salva curvas de perda e AUC ao longo das épocas."""
    hist = history.history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(hist["loss"], label="treino")
    ax1.plot(hist["val_loss"], label="validação")
    ax1.set_title("Perda (binary crossentropy)")
    ax1.set_xlabel("época"); ax1.legend()
    ax2.plot(hist["auc"], label="treino")
    ax2.plot(hist["val_auc"], label="validação")
    ax2.set_title("AUC")
    ax2.set_xlabel("época"); ax2.legend()
    fig.tight_layout()
    out = config.FIGURES_DIR / "historico_treino.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"[avaliação] figura salva: {out}")


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["falso positivo", "confirmado"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusão (teste)")
    fig.tight_layout()
    out = config.FIGURES_DIR / "matriz_confusao.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"[avaliação] figura salva: {out}")


def plot_roc(y_true: np.ndarray, y_score: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("taxa de falsos positivos")
    ax.set_ylabel("taxa de verdadeiros positivos")
    ax.set_title("Curva ROC (teste)")
    ax.legend()
    fig.tight_layout()
    out = config.FIGURES_DIR / "curva_roc.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"[avaliação] figura salva: {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Treina a MLP de detecção de exoplanetas.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args(argv)

    # 1) Dados
    df = collect(force=args.force_download)
    data = prepare(df)

    # 2) Pesos de classe (há mais falsos positivos que confirmados)
    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=data.y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"[treino] class_weight = {class_weight}")

    # 3) Modelo
    model = build_model(input_dim=data.X_train.shape[1])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=12, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
        ),
    ]

    history = model.fit(
        data.X_train, data.y_train,
        validation_data=(data.X_val, data.y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    # 4) Avaliação no conjunto de teste
    y_score = model.predict(data.X_test, verbose=0).ravel()
    y_pred = (y_score >= 0.5).astype(int)

    print("\n=== Relatório de classificação (teste) ===")
    report = classification_report(
        data.y_test, y_pred,
        target_names=["falso positivo", "confirmado"],
        digits=3,
    )
    print(report)
    test_auc = roc_auc_score(data.y_test, y_score)
    print(f"AUC (teste): {test_auc:.3f}")

    # 5) Figuras e artefatos
    plot_history(history)
    plot_confusion(data.y_test, y_pred)
    plot_roc(data.y_test, y_score)

    model_path = config.MODELS_DIR / "exoplanet_mlp.keras"
    model.save(model_path)
    print(f"[treino] modelo salvo: {model_path}")

    metrics = {
        "test_auc": float(test_auc),
        "n_features": int(data.X_train.shape[1]),
        "features": data.feature_names,
        "report": report,
    }
    with open(config.MODELS_DIR / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
