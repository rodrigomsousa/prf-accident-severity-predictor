# CLAUDE.md

Contexto para assistentes de IA trabalhando neste repositório.

## Sobre o projeto

Projeto de estudo e portfólio em Data Science. O autor é programador experiente, mas iniciante em DS: **explique os conceitos de DS** (métricas, leakage, validação, escolha de modelos) de forma didática junto com o código. Idioma: português.

**Escopo principal:** modelo de triagem para o 191 da PRF, que estima no momento da ligação a probabilidade de o acidente ter vítimas graves ou fatais, para apoiar o despacho de resgate avançado. A abordagem preventiva (só variáveis pré-evento) é um **experimento comparativo** no notebook, não o modelo principal.

## Decisões já tomadas (não revisitar sem motivo)

- **Alvo binário** `grave` = ao menos um morto ou ferido grave (~28%). Multiclasse pode ser considerada no futuro.
- **Validação temporal:** treino 2024 + 2025, teste 2026 (ano parcial, até agosto). Nunca usar split aleatório.
- **Leakage:** as colunas de desfecho em `LEAKAGE_COLS` nunca entram como features. `causa_acidente` também fica fora da triagem, porque só é definida na investigação.
- **Conjuntos de features** em `src/data_preprocessing.py`: `TRIAGE` (contexto + `tipo_acidente`, `veiculos`, `pessoas` + agregados de veículos/pessoas) e `PREVENTIVE` (só contexto, incluindo localização). `municipio` e `trecho` (BR + UF + faixa de 10 km) passam por `TargetEncoder` com cross-fitting dentro do pipeline; nunca calcular taxas por categoria fora do pipeline (leakage).
- Métricas principais: **ROC-AUC e PR-AUC** (a base é desbalanceada e a acurácia engana). Sempre comparar com o baseline no mesmo split e relatar números reais.

## Estado atual e próximos passos

Resultados da triagem no teste 2026 (ROC-AUC / PR-AUC), com features geográficas e de veículos/pessoas: Logistic Regression (`class_weight="balanced"`) 0.757 / 0.551; Random Forest 0.764 / 0.584; **XGBoost 0.774 / 0.599** (melhor). Preventivo (Logistic Regression): 0.633 / 0.401. Recall/precision em `train.py` usam o limiar padrão 0.5.

Histórico dos ganhos: sem nenhuma das duas, XGBoost 0.718 / 0.534. Features geográficas: +0.005 de PR-AUC (IC 95% por bootstrap pareado +0.003 a +0.008); trechos de 1/5 km, volume histórico do trecho e delegacia ficaram dentro do ruído. **Features de veículos/pessoas** (`n_moto`, `n_bicicleta`, `n_pesado`, `n_onibus`, `tem_pedestre`, de `data/raw/pessoas{ano}.csv`, contagem por `id_veiculo`, teto 5): o maior ganho, +0.06 de PR-AUC no XGBoost (IC 95% +0.056 a +0.065), e as árvores passaram a se separar do modelo linear. `estado_fisico` e demais colunas de desfecho dos arquivos por pessoa são leakage; idade e sexo foram deixados de fora (o solicitante raramente sabe; idade tem valores inválidos). `TRIAGE_BASE` é o conjunto sem veículos/pessoas, mantido para comparações.

Limiar ajustado (`src/threshold.py`): escolhido sem tocar no teste (ajusta em 2024, escolhe em 2025 o maior limiar com recall >= 80%), resultou em 0.391. No teste 2026 o XGBoost fica com recall 0.825, precision 0.408 e aciona resgate em 56.7% das ligações (contra 0.673 / 0.487 / 38.7% com 0.5; antes das features de veículos eram 64.2% de acionamento).

**Ressalva importante:** os arquivos por pessoa são o registro final da PRF, não o relato da ligação. O ganho em produção tende a ser menor se o relato for incompleto.

README e notebook estão atualizados (o notebook tem as seções 4.2 geográficas, 4.3 veículos e pessoas, e 4.4 limiar).

Próximos passos:
1. Teste de robustez ao relato: esconder aleatoriamente parte das informações de veículos/pessoas no teste e medir quanto do ganho sobrevive.
2. Multiclasse (leve, grave, fatal), se fizer sentido.

## Comandos

```bash
venv/Scripts/python src/train.py                    # treina, avalia e salva models/{logistic_regression,random_forest,xgboost}.joblib
venv/Scripts/python src/threshold.py                # escolhe o limiar do XGBoost (validação 2025) e salva models/xgboost_threshold.json
cd notebooks && ../venv/Scripts/jupyter nbconvert --to notebook --execute --inplace 01_eda_and_modeling.ipynb
```

- Dados em `data/raw/datatran{ano}.csv` e `pessoas{ano}.csv` (separador `;`, encoding `latin1`, decimal `,`), ignorados pelo git.
- Python 3.14, pandas 3, scikit-learn 1.9 (ex.: `RocCurveDisplay` usa `curve_kwargs`, não `color=`).

## Convenções

- **Notebook:** editar o `.ipynb` diretamente. O nbconvert grava timestamps em `metadata.execution` de cada célula, e o VS Code os remove ao abrir. **Limpe esses metadados antes de commitar** para não gerar diffs espúrios.
- Commits em português. A branch padrão é `main`.
