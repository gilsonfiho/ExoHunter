"""Coleta de dados do NASA Exoplanet Archive via serviço TAP.

Baixa a tabela cumulativa de KOIs (Kepler Objects of Interest) e salva em
``data/raw/koi_cumulative.csv``. A API é aberta e não exige chave.

Uso:
    python -m src.data_collection            # baixa (usa cache se já existir)
    python -m src.data_collection --force    # re-baixa mesmo com cache
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd
import requests

from . import config


def build_query() -> str:
    """Monta a query ADQL para o serviço TAP.

    Seleciona as colunas de identificação, a coluna-alvo e as features.
    """
    cols = config.ID_COLS + [config.TARGET_COL] + config.FEATURES
    col_list = ", ".join(cols)
    return f"select {col_list} from {config.TAP_TABLE}"


def fetch_koi(timeout: int = 120) -> pd.DataFrame:
    """Consulta o TAP e devolve o resultado como DataFrame."""
    query = build_query()
    params = {"query": query, "format": "csv"}
    print(f"[coleta] Consultando NASA Exoplanet Archive (tabela '{config.TAP_TABLE}')...")
    resp = requests.get(config.TAP_URL, params=params, timeout=timeout)
    resp.raise_for_status()

    # Respostas de erro do TAP vêm como texto, não CSV válido.
    if resp.text.lstrip().upper().startswith("ERROR"):
        raise RuntimeError(f"TAP retornou erro:\n{resp.text[:500]}")

    from io import StringIO

    df = pd.read_csv(StringIO(resp.text))
    print(f"[coleta] Recebidas {len(df):,} linhas e {df.shape[1]} colunas.")
    return df


def collect(force: bool = False) -> pd.DataFrame:
    """Baixa os dados (ou usa o cache em disco) e devolve o DataFrame."""
    if config.RAW_KOI_CSV.exists() and not force:
        print(f"[coleta] Usando cache: {config.RAW_KOI_CSV}")
        return pd.read_csv(config.RAW_KOI_CSV)

    df = fetch_koi()
    df.to_csv(config.RAW_KOI_CSV, index=False)
    print(f"[coleta] Salvo em {config.RAW_KOI_CSV}")
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Coleta KOIs do NASA Exoplanet Archive.")
    parser.add_argument("--force", action="store_true", help="re-baixar ignorando cache")
    args = parser.parse_args(argv)

    df = collect(force=args.force)
    # Resumo rápido da distribuição de classes.
    print("\n[coleta] Distribuição de", config.TARGET_COL, ":")
    print(df[config.TARGET_COL].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
