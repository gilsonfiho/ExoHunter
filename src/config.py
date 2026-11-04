"""Configuração central do projeto ExoHunter.

Caminhos, parâmetros da coleta de dados e features usadas pelo modelo.
Mantemos tudo num único lugar para o pipeline ficar fácil de ajustar.
"""
from pathlib import Path

# --- Caminhos -------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

for _d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_KOI_CSV = DATA_RAW / "koi_cumulative.csv"

# --- Fonte de dados: NASA Exoplanet Archive (serviço TAP) -----------------
# API aberta, sem necessidade de chave. Documentação:
# https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html
TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
TAP_TABLE = "cumulative"  # tabela cumulativa de Kepler Objects of Interest (KOI)

# Coluna-alvo (disposição definida pelo arquivo da NASA)
TARGET_COL = "koi_disposition"

# Features físicas do trânsito e da estrela hospedeira.
# OBS: deixamos DE FORA de propósito 'koi_score' e os flags 'koi_fpflag_*',
# porque derivam diretamente da vetagem que decide o rótulo (vazamento de
# informação / data leakage). Queremos que a rede aprenda a partir da física.
FEATURES = [
    # --- Parâmetros do trânsito ---
    "koi_period",      # período orbital (dias)
    "koi_duration",    # duração do trânsito (horas)
    "koi_depth",       # profundidade do trânsito (ppm)
    "koi_impact",      # parâmetro de impacto
    "koi_prad",        # raio planetário (raios da Terra)
    "koi_teq",         # temperatura de equilíbrio (K)
    "koi_insol",       # fluxo de insolação (relativo à Terra)
    "koi_model_snr",   # razão sinal-ruído do trânsito
    # --- Parâmetros da estrela hospedeira ---
    "koi_steff",       # temperatura efetiva estelar (K)
    "koi_slogg",       # gravidade superficial (log g)
    "koi_srad",        # raio estelar (raios solares)
    "koi_kepmag",      # magnitude na banda Kepler
]

# Colunas extras úteis para identificação/EDA (não entram no modelo)
ID_COLS = ["kepoi_name", "kepler_name", "koi_pdisposition", "koi_score"]

# Classes que usamos no problema binário (descartamos CANDIDATE, sem rótulo
# definitivo). 1 = exoplaneta confirmado, 0 = falso positivo.
POSITIVE_CLASS = "CONFIRMED"
NEGATIVE_CLASS = "FALSE POSITIVE"

# --- Hiperparâmetros / reprodutibilidade ----------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2  # fração do treino reservada para validação
