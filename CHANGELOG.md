# Changelog

Todas as mudanças notáveis deste projeto são documentadas neste arquivo.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/)
e o projeto adota [Versionamento Semântico](https://semver.org/lang/pt-BR/).

## [0.2.1] - 2026-11-04
### Adicionado
- Gráficos de histórico de treino (acurácia/AUC) e curva ROC embutidos no
  README, com cópias versionadas das figuras em `docs/img/`.

## [0.2.0] - 2026-11-04
### Adicionado
- Script de predição que classifica todos os KOIs — inclusive os candidatos
  sem veredito — e salva `data/processed/predictions.csv` (`src/predict.py`).
- Dashboard em Streamlit (`app.py`) com visão geral do processo, tabela
  filtrável de planetas classificados e aba de candidatos promissores.
- Tema visual escuro (`.streamlit/config.toml`); `streamlit` e `plotly` nas
  dependências.

### Corrigido
- Ordem de importação em `src/predict.py`: TensorFlow é importado antes de
  `joblib`/`numpy`/`pandas` para evitar falha de carregamento de DLL no Windows.

## [0.1.0] - 2026-11-03
### Adicionado
- Coleta de KOIs do NASA Exoplanet Archive via API TAP (`src/data_collection.py`).
- Pré-processamento com split estratificado e padronização (`src/preprocessing.py`).
- Rede neural (Keras/TensorFlow) para classificar exoplanetas confirmados
  vs. falsos positivos (`src/model.py`, `src/train.py`).
- Avaliação com AUC, matriz de confusão e curva ROC; artefatos salvos em
  `models/` e `reports/figures/`.
- Documentação do projeto no README.

## [0.0.0] - 2026-11-03
### Adicionado
- Estrutura inicial do projeto: diretórios, dependências e `.gitignore`.
