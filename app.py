"""Dashboard ExoHunter — interface para acompanhar o processo e explorar
todos os KOIs analisados e classificados pelo modelo.

Executar com:
    streamlit run app.py

Lê apenas artefatos já gerados (não importa TensorFlow):
    - data/processed/predictions.csv  (via `python -m src.predict`)
    - models/metrics.json             (via `python -m src.train`)
    - reports/figures/*.png           (opcional, gerado no treino)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# --- Caminhos -------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
PRED_CSV = ROOT / "data" / "processed" / "predictions.csv"
METRICS_JSON = ROOT / "models" / "metrics.json"
FIGURES_DIR = ROOT / "reports" / "figures"

POS, NEG, CAND = "CONFIRMED", "FALSE POSITIVE", "CANDIDATE"
LABEL_PT = {POS: "Confirmado", NEG: "Falso positivo", CAND: "Candidato"}
CORES = {"Confirmado": "#39d98a", "Falso positivo": "#ff5c7c", "Candidato": "#f5c451"}

st.set_page_config(page_title="ExoHunter", page_icon="🪐", layout="wide")


# --- Carregamento de dados (com cache) ------------------------------------
@st.cache_data(show_spinner=False)
def carregar_predicoes() -> pd.DataFrame | None:
    if not PRED_CSV.exists():
        return None
    df = pd.read_csv(PRED_CSV)
    df["disposicao_pt"] = df["koi_disposition"].map(LABEL_PT)
    df["classificacao_pt"] = df["classificacao_modelo"].map(LABEL_PT)
    return df


@st.cache_data(show_spinner=False)
def carregar_metricas() -> dict:
    if METRICS_JSON.exists():
        return json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    return {}


# --- Cabeçalho ------------------------------------------------------------
st.title("🪐 ExoHunter")
st.caption(
    "Detecção de exoplanetas do telescópio Kepler com deep learning · "
    "dados do [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)"
)

df = carregar_predicoes()
metrics = carregar_metricas()

if df is None:
    st.error(
        "Predições não encontradas. Rode o pipeline primeiro:\n\n"
        "```\npython -m src.train\npython -m src.predict\n```"
    )
    st.stop()

# --- Indicadores principais ----------------------------------------------
n_total = len(df)
n_conf = int((df["koi_disposition"] == POS).sum())
n_fp = int((df["koi_disposition"] == NEG).sum())
n_cand = int((df["koi_disposition"] == CAND).sum())
rotulados = df[df["acertou"].notna()]
acuracia = rotulados["acertou"].mean() if len(rotulados) else float("nan")
auc = metrics.get("test_auc")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("KOIs analisados", f"{n_total:,}".replace(",", "."))
c2.metric("Confirmados (reais)", f"{n_conf:,}".replace(",", "."))
c3.metric("Falsos positivos (reais)", f"{n_fp:,}".replace(",", "."))
c4.metric("Candidatos", f"{n_cand:,}".replace(",", "."))
c5.metric("AUC do modelo", f"{auc:.3f}" if auc else "—",
          help="Área sob a curva ROC no conjunto de teste.")

st.divider()

tab_proc, tab_planetas, tab_cand = st.tabs(
    ["📊 Visão geral do processo", "🪐 Planetas classificados", "✨ Candidatos promissores"]
)

# ========================================================================
# TAB 1 — Visão geral do processo
# ========================================================================
with tab_proc:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Distribuição real (NASA)")
        dist_real = (
            df["disposicao_pt"].value_counts().rename_axis("classe").reset_index(name="qtd")
        )
        fig = px.pie(dist_real, names="classe", values="qtd", hole=0.45,
                     color="classe", color_discrete_map=CORES)
        fig.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10), height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Classificação do modelo")
        dist_mod = (
            df["classificacao_pt"].value_counts().rename_axis("classe").reset_index(name="qtd")
        )
        fig = px.pie(dist_mod, names="classe", values="qtd", hole=0.45,
                     color="classe", color_discrete_map=CORES)
        fig.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10), height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Desempenho do modelo (nos KOIs rotulados)")
    mc1, mc2 = st.columns([1, 1])

    with mc1:
        if len(rotulados):
            matriz = pd.crosstab(
                rotulados["disposicao_pt"], rotulados["classificacao_pt"]
            ).reindex(index=["Falso positivo", "Confirmado"],
                      columns=["Falso positivo", "Confirmado"], fill_value=0)
            fig = px.imshow(
                matriz, text_auto=True, aspect="auto",
                labels=dict(x="Previsto", y="Real", color="KOIs"),
                color_continuous_scale="Blues",
            )
            fig.update_layout(margin=dict(t=30, b=10, l=10, r=10), height=340,
                              title="Matriz de confusão")
            st.plotly_chart(fig, use_container_width=True)

    with mc2:
        st.metric("Acurácia (rotulados)",
                  f"{acuracia:.1%}" if acuracia == acuracia else "—")
        st.metric("AUC (teste)", f"{auc:.3f}" if auc else "—")
        hist_png = FIGURES_DIR / "historico_treino.png"
        if hist_png.exists():
            st.image(str(hist_png), caption="Histórico de treino", use_column_width=True)

    with st.expander("ℹ️ Como funciona"):
        st.markdown(
            "O Kepler mede o brilho de estrelas e procura **trânsitos** — quedas "
            "periódicas de luz quando um planeta passa à frente da estrela. Cada "
            "sinal vira um *Kepler Object of Interest* (KOI). A rede neural usa os "
            "parâmetros físicos do trânsito e da estrela para prever se o KOI é um "
            "**exoplaneta confirmado** ou um **falso positivo** — e aplicamos isso "
            "também aos **candidatos** que ainda não têm veredito."
        )

# ========================================================================
# Filtros (barra lateral) — usados nas abas de planetas
# ========================================================================
st.sidebar.header("🔎 Filtros")
busca = st.sidebar.text_input("Buscar por nome (KOI ou Kepler)")
disp_sel = st.sidebar.multiselect(
    "Disposição real", options=list(LABEL_PT.values()),
    default=list(LABEL_PT.values()),
)
mod_sel = st.sidebar.multiselect(
    "Classificação do modelo", options=["Confirmado", "Falso positivo"],
    default=["Confirmado", "Falso positivo"],
)
prob_min, prob_max = st.sidebar.slider(
    "Probabilidade de ser exoplaneta", 0.0, 1.0, (0.0, 1.0), 0.01
)

filtrado = df[
    df["disposicao_pt"].isin(disp_sel)
    & df["classificacao_pt"].isin(mod_sel)
    & df["prob_exoplaneta"].between(prob_min, prob_max)
].copy()
if busca:
    b = busca.strip().lower()
    filtrado = filtrado[
        filtrado["kepoi_name"].str.lower().str.contains(b, na=False)
        | filtrado["kepler_name"].str.lower().str.contains(b, na=False)
    ]

COLS_EXIBE = {
    "kepoi_name": "KOI",
    "kepler_name": "Nome Kepler",
    "disposicao_pt": "Disposição real",
    "classificacao_pt": "Classificação modelo",
    "prob_exoplaneta": "Prob. exoplaneta",
    "koi_period": "Período (dias)",
    "koi_prad": "Raio (R⊕)",
    "koi_teq": "Temp. eq. (K)",
    "koi_model_snr": "SNR",
}
CONFIG_COLS = {
    "Prob. exoplaneta": st.column_config.ProgressColumn(
        "Prob. exoplaneta", min_value=0.0, max_value=1.0, format="%.2f"
    ),
    "Período (dias)": st.column_config.NumberColumn(format="%.2f"),
    "Raio (R⊕)": st.column_config.NumberColumn(format="%.2f"),
    "Temp. eq. (K)": st.column_config.NumberColumn(format="%.0f"),
    "SNR": st.column_config.NumberColumn(format="%.1f"),
}


def tabela(frame: pd.DataFrame) -> None:
    exibe = frame[list(COLS_EXIBE)].rename(columns=COLS_EXIBE)
    st.dataframe(exibe, use_container_width=True, hide_index=True,
                 column_config=CONFIG_COLS, height=460)


# ========================================================================
# TAB 2 — Planetas classificados
# ========================================================================
with tab_planetas:
    st.subheader(f"{len(filtrado):,} KOIs".replace(",", ".") + " após filtros")

    fig = px.scatter(
        filtrado, x="koi_period", y="koi_prad", color="classificacao_pt",
        color_discrete_map=CORES, log_x=True, log_y=True,
        hover_name="kepoi_name",
        labels={"koi_period": "Período orbital (dias, log)",
                "koi_prad": "Raio planetário (R⊕, log)",
                "classificacao_pt": "Modelo"},
        opacity=0.6,
    )
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=380)
    st.plotly_chart(fig, use_container_width=True)

    tabela(filtrado.sort_values("prob_exoplaneta", ascending=False))

    st.download_button(
        "⬇️ Baixar seleção (CSV)",
        data=filtrado.to_csv(index=False).encode("utf-8"),
        file_name="exohunter_selecao.csv", mime="text/csv",
    )

# ========================================================================
# TAB 3 — Candidatos promissores
# ========================================================================
with tab_cand:
    st.subheader("Candidatos que o modelo classifica como exoplanetas")
    cand = df[(df["koi_disposition"] == CAND) & (df["classificacao_modelo"] == POS)]
    cand = cand.sort_values("prob_exoplaneta", ascending=False)
    st.markdown(
        f"Dos **{n_cand:,}** candidatos sem veredito, o modelo aponta "
        f"**{len(cand):,}** como prováveis exoplanetas "
        f"(probabilidade ≥ 0,5).".replace(",", ".")
    )
    tabela(cand)
    st.caption(
        "⚠️ São previsões do modelo, não confirmações oficiais — mas é "
        "exatamente aqui que uma triagem automática ajuda a priorizar alvos."
    )
