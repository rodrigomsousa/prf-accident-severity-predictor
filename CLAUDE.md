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

**Robustez ao relato (feito):** degradando só o teste 2026 (com prob. p a ligação perde o bloco de veículos/pessoas, omitido = 0), o XGBoost treinado com relato completo perde o ganho quase linearmente: a PR-AUC empata com o piso (`TRIAGE_BASE`, 0.539) em ~50% de omissão e fica pior acima disso, e o recall do limiar 0.391 cai de 0.825 para 0.61 (50%) e 0.40 (100%). A defesa é treinar com omissão simulada (`aug-nan`: cada acidente com taxa p ~ U(0,1), bloco omitido = `NaN`): com relato completo 0.770 / 0.597 (perde só 0.002 de PR-AUC), 0.536 de PR-AUC sem nenhuma informação e recall ~0.82 em qualquer nível. O limiar do `aug-nan` é 0.389 (validação 2025 também degradada); limiares ótimos por nível ficaram em 0.388 a 0.394, então um único serve. Preço: sem o relato, precision 0.40 -> 0.36 e acionamento 58% -> 64%. Omitir com 0 (em vez de `NaN`) no treino é pior (`aug-zero`). Limite: a omissão é simulada, aleatória e igual no treino e no teste; omissão real pode correlacionar com gravidade.

**Cuidado ao augmentar o treino:** nunca empilhar uma cópia degradada ao treino original. A mesma linha cairia em folds diferentes do `TargetEncoder` e vazaria o alvo; cada acidente aparece uma vez, com taxa de omissão própria.

`XGBoost (aug-nan)` é treinado e salvo por `train.py` (`models/xgboost_aug_nan.joblib`) junto com o XGBoost completo, que continua como comparação. A inferência aceita `NaN` nas colunas de `PEOPLE_COLS` para "não informado". O modelo principal para uso em produção é o `aug-nan`, com limiar de `models/xgboost_nan_threshold.json`.

README e notebook estão atualizados (o notebook tem as seções 4.2 geográficas, 4.3 veículos e pessoas, 4.4 limiar e 4.5 robustez ao relato incompleto).

Próximos passos:
1. Multiclasse (leve, grave, fatal), se fizer sentido.
2. Outras informações que o atendente pode perguntar (veículo preso nas ferragens, carga perigosa), se houver dados que as aproximem.
3. Validar a hipótese de omissão aleatória com relatos reais das ligações, se houver acesso a eles.

## Comandos

```bash
venv/Scripts/python src/train.py                    # treina, avalia e salva models/{logistic_regression,random_forest,xgboost,xgboost_aug_nan}.joblib
venv/Scripts/python src/threshold.py                # escolhe o limiar do XGBoost (validação 2025) e salva models/xgboost_threshold.json
venv/Scripts/python src/threshold_nan.py            # limiar do aug-nan (validação 2025 degradada) em models/xgboost_nan_threshold.json
venv/Scripts/python src/robustness.py               # curva de degradação do modelo completo (dados em data/robustness_results.csv)
venv/Scripts/python src/augmentation.py             # completo vs aug-zero vs aug-nan sob omissão (data/augmentation_results.csv)
cd notebooks && ../venv/Scripts/jupyter nbconvert --to notebook --execute --inplace 01_eda_and_modeling.ipynb
```

- Dados em `data/raw/datatran{ano}.csv` e `pessoas{ano}.csv` (separador `;`, encoding `latin1`, decimal `,`), ignorados pelo git.
- Python 3.14, pandas 3, scikit-learn 1.9 (ex.: `RocCurveDisplay` usa `curve_kwargs`, não `color=`).

## Convenções

- **Notebook:** editar o `.ipynb` diretamente. O nbconvert grava timestamps em `metadata.execution` de cada célula, e o VS Code os remove ao abrir. **Limpe esses metadados antes de commitar** para não gerar diffs espúrios.
- Commits em português. A branch padrão é `main`.
