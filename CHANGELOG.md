# Changelog

Todas as mudanças notáveis deste projeto são documentadas neste arquivo.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/)
e o projeto adota [Versionamento Semântico](https://semver.org/lang/pt-BR/).

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
