# CLAUDE.md

Contexto para assistentes de IA trabalhando neste repositório.

## Sobre o projeto

Projeto de estudo e portfólio em Data Science. O autor é programador experiente, mas iniciante em DS: **explique os conceitos de DS** (métricas, leakage, validação, escolha de modelos) de forma didática junto com o código. Idioma: português.

**Escopo principal:** modelo de triagem para o 191 da PRF, que estima no momento da ligação a probabilidade de o acidente ter vítimas graves ou fatais, para apoiar o despacho de resgate avançado. A abordagem preventiva (só variáveis pré-evento) é um **experimento comparativo** no notebook, não o modelo principal.

## Decisões já tomadas (não revisitar sem motivo)

- **Alvo binário** `grave` = ao menos um morto ou ferido grave (~28%). Multiclasse pode ser considerada no futuro.
- **Validação temporal:** treino 2024 + 2025, teste 2026 (ano parcial, até agosto). Nunca usar split aleatório.
- **Leakage:** as colunas de desfecho em `LEAKAGE_COLS` nunca entram como features. `causa_acidente` também fica fora da triagem, porque só é definida na investigação.
- **Conjuntos de features** em `src/data_preprocessing.py`: `TRIAGE` (contexto + `tipo_acidente`, `veiculos`, `pessoas`) e `PREVENTIVE` (só contexto, incluindo localização). `municipio` e `trecho` (BR + UF + faixa de 10 km) passam por `TargetEncoder` com cross-fitting dentro do pipeline; nunca calcular taxas por categoria fora do pipeline (leakage).
- Métricas principais: **ROC-AUC e PR-AUC** (a base é desbalanceada e a acurácia engana). Sempre comparar com o baseline no mesmo split e relatar números reais.

## Estado atual e próximos passos

Resultados da triagem no teste 2026, com features geográficas (ROC-AUC / PR-AUC): Logistic Regression (`class_weight="balanced"`) 0.715 / 0.524; Random Forest 0.717 / 0.530; **XGBoost 0.721 / 0.539** (melhor, por pouco). Preventivo (Logistic Regression): 0.633 / 0.401. Os modelos de árvore quase não superam o linear, então o gargalo são as features, não o algoritmo. Recall/precision em `train.py` usam o limiar padrão 0.5.

Features geográficas: ganho real mas pequeno (XGBoost +0.005 de PR-AUC, IC 95% por bootstrap pareado de +0.003 a +0.007; antes era 0.718 / 0.534). Trechos de 1/5 km, volume histórico do trecho e delegacia foram testados e ficaram dentro do ruído, então ficou a versão mais simples. O preventivo ganhou mais (+0.014 de PR-AUC), porque na triagem `tipo_acidente` já carrega a maior parte do sinal.

Limiar ajustado (`src/threshold.py`): escolhido sem tocar no teste (ajusta em 2024, escolhe em 2025 o maior limiar com recall >= 80%), resultou em 0.389. No teste 2026 o XGBoost fica com recall 0.825, precision 0.360 e aciona resgate em 64.2% das ligações (contra 0.592 / 0.454 / 36.5% com 0.5). O custo alto confirma que faltam features discriminativas.

README e notebook estão atualizados com esses resultados (o notebook tem as seções 4.2, features geográficas com bootstrap, e 4.3, limiar).

Próximo passo:
1. Features de veículos/pessoas envolvidas (ex.: moto, pedestre), dos dados da PRF por pessoa/veículo, como aproximação do que o 191 pode perguntar na ligação.

## Comandos

```bash
venv/Scripts/python src/train.py                    # treina, avalia e salva models/{logistic_regression,random_forest,xgboost}.joblib
venv/Scripts/python src/threshold.py                # escolhe o limiar do XGBoost (validação 2025) e salva models/xgboost_threshold.json
cd notebooks && ../venv/Scripts/jupyter nbconvert --to notebook --execute --inplace 01_eda_and_modeling.ipynb
```

- Dados em `data/raw/datatran{ano}.csv` (separador `;`, encoding `latin1`, decimal `,`), ignorados pelo git.
- Python 3.14, pandas 3, scikit-learn 1.9 (ex.: `RocCurveDisplay` usa `curve_kwargs`, não `color=`).

## Convenções

- **Notebook:** editar o `.ipynb` diretamente. O nbconvert grava timestamps em `metadata.execution` de cada célula, e o VS Code os remove ao abrir. **Limpe esses metadados antes de commitar** para não gerar diffs espúrios.
- Commits em português. A branch padrão é `main`.
