# 🚗 PRF Accident Severity Predictor

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Modelo de triagem inteligente para o sistema de emergência 191 da PRF, para apoiar a decisão de despacho de resgate avançado.**

O projeto usa os dados abertos da **Polícia Rodoviária Federal (PRF)** sobre acidentes nas rodovias federais brasileiras.

---

## 🎯 Objetivo do Projeto

Quando uma ocorrência chega pelo **191**, o atendente precisa decidir rapidamente se envia **resgate avançado**. O modelo estima, no momento da ligação, a **probabilidade de o acidente ter vítimas graves ou fatais**. Para isso, usa apenas informações disponíveis naquele instante:

* **Contexto:** local (UF, BR, km, município, coordenadas), data e horário, condição meteorológica e características da via.
* **Relato do solicitante:** tipo de acidente, número de veículos e de pessoas envolvidas, e quais veículos (moto, bicicleta, caminhão, ônibus) e se há pedestre.

Como experimento complementar, o notebook avalia uma **abordagem preventiva**: prever a gravidade só com variáveis pré-evento, para apoiar o planejamento do patrulhamento.

---

## 📊 Dataset & Dados Abertos

Dados do portal de **Dados Abertos da PRF** (agrupados por ocorrência e por pessoa): 2024, 2025 e 2026 (até setembro), cerca de 200 mil acidentes e 524 mil pessoas envolvidas.

* **Variável alvo (`grave`):** binária. Vale 1 se houve ao menos um **morto ou ferido grave** (~28% dos acidentes).
* **Prevenção de *data leakage*:** as colunas que descrevem o desfecho (mortos, feridos, ilesos, `classificacao_acidente`) não são usadas como features. `causa_acidente` também fica de fora, porque é definida na investigação, depois do atendimento.
* **Validação temporal:** o modelo treina com **2024 + 2025** e é testado em **2026**, simulando o uso real em ocorrências futuras.
* **Histórico do local:** município e trecho da rodovia (BR + UF + faixa de 10 km) têm milhares de categorias e viram a taxa histórica de acidentes graves de cada uma (*target encoding*). A codificação é feita dentro do pipeline, com *cross-fitting*: no treino, cada acidente recebe a taxa calculada sem ele mesmo, senão o alvo vazaria para a feature.

---

## 🛠️ Arquitetura do Projeto & Tecnologias

O projeto usa **Pipelines do Scikit-Learn** (pré-processamento + modelo em um único objeto) para evitar *data leakage* e garantir reprodutibilidade.

* **Linguagem:** Python
* **Manipulação de Dados:** Pandas, NumPy
* **Visualização:** Matplotlib
* **Machine Learning:** Scikit-Learn, XGBoost
* **Persistência de Modelo:** Joblib

---

## 🚀 Estrutura do Repositório

```text
prf-accident-severity-predictor/
│
├── data/raw/              # CSVs da PRF: datatran{ano}.csv e pessoas{ano}.csv (ignorado no git)
├── models/                # Pipelines treinados (.joblib) e limiares (.json)
├── reports/               # Gráficos gerados pelos scripts (SHAP)
├── notebooks/
│   └── 01_eda_and_modeling.ipynb   # EDA, modelagem, robustez e experimento preventivo
├── src/
│   ├── data_preprocessing.py       # Carga, alvo, features e conjuntos de features
│   ├── train.py                    # Pipeline, treino, avaliação e persistência
│   ├── threshold.py                # Ajuste do limiar de decisão priorizando recall
│   ├── degradation.py              # Simulação de relato incompleto (omissão de veículos/pessoas)
│   ├── robustness.py               # Teste de robustez do modelo ao relato incompleto
│   ├── augmentation.py             # Compara treinar com e sem omissão simulada
│   ├── threshold_nan.py            # Limiar do modelo treinado com omissão
│   ├── calibration.py              # Calibração das probabilidades (Platt e isotônica)
│   ├── calibration_prod.py         # Calibrador do modelo de produção (Platt OOF)
│   ├── subgroups.py                # Recall/precision por subgrupo
│   ├── group_threshold.py          # Limiares por grupo vs limiar único
│   ├── cost_analysis.py            # Limiar ótimo por custo e custo implícito do limiar atual
│   ├── drift.py                    # Avaliação rolante mês a mês (congelado vs retreinado)
│   ├── informative_omission.py     # Omissão do relato que depende da ocorrência
│   └── explain.py                  # Interpretabilidade com valores SHAP
├── app/
│   └── streamlit_app.py            # Demo: probabilidade, decisão e SHAP de uma ocorrência
├── tests/                          # Testes (leakage, degradação, pipeline, modelo de produção)
├── pytest.ini
├── .gitignore
├── readme.md
└── requirements.txt
```

---

## 📈 Resultados e Desempenho

Teste em **2026** (54.895 acidentes, jan–set, 28% graves):

| Modelo | Acurácia | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (Dummy, sempre "leve") | 0.721 | 0.000 | 0.000 | 0.000 | 0.500 | 0.279 |
| Triagem: Logistic Regression | 0.705 | 0.479 | 0.647 | 0.550 | 0.756 | 0.549 |
| Triagem: Random Forest | 0.746 | 0.546 | 0.539 | 0.542 | 0.764 | 0.581 |
| **Triagem: XGBoost** | 0.710 | 0.486 | 0.671 | 0.563 | **0.773** | **0.596** |
| Triagem: XGBoost treinado com omissão (relato completo no teste) | 0.708 | 0.483 | 0.670 | 0.561 | 0.770 | 0.594 |

Precision, recall e F1 desta tabela usam o limiar padrão de 0.5. A última linha é o modelo preparado para relatos incompletos (veja a seção de robustez abaixo): com relato completo, perde só 0.002 de PR-AUC. Com os meses de jul–set incluídos no teste, os números mal mudaram em relação ao corte até agosto (XGBoost 0.774 / 0.599 antes, 0.773 / 0.596 agora), o que indica estabilidade.

**O que mais melhorou o modelo foram as features de veículos e pessoas** (quantas motos, bicicletas, caminhões e ônibus, e se há pedestre), agregadas dos arquivos da PRF por pessoa. Sem elas, os três algoritmos ficavam colados em ROC-AUC ≈ 0.72. O XGBoost foi de 0.722 / 0.537 para 0.773 / 0.596 (ROC-AUC / PR-AUC), com intervalo de 95% do ganho em PR-AUC entre +0.055 e +0.064 (bootstrap pareado no teste). Com isso, as árvores passaram a se separar do modelo linear. Apenas o `estado_fisico` e demais colunas de desfecho dos arquivos por pessoa ficam de fora.

As features geográficas (coordenadas e histórico de município e trecho) trouxeram um ganho menor, mas estatisticamente real: +0.005 de PR-AUC, com intervalo de 95% entre +0.003 e +0.007.

> **Ressalva:** os arquivos por pessoa são o registro final da PRF. O modelo supõe que o atendente consegue saber na ligação se há moto ou pedestre. A seção de robustez abaixo mede o que acontece quando o relato é incompleto.

### Ajuste do limiar de decisão

O modelo devolve uma probabilidade, e o limiar decide a partir de qual valor o acidente recebe resgate avançado. Deixar de enviar resgate a um acidente grave custa mais caro do que um envio desnecessário, então o limiar foi ajustado para garantir **recall de pelo menos 80%**.

O limiar foi escolhido **sem olhar o teste**: o modelo treina em 2024, o limiar é escolhido em 2025 e só então o modelo é retreinado em 2024 + 2025 e avaliado em 2026.

| XGBoost no teste 2026 | Limiar | Recall | Precision | Acionamentos |
| :--- | :---: | :---: | :---: | :---: |
| Limiar padrão | 0.500 | 0.671 | 0.486 | 38.5% |
| **Limiar ajustado** | **0.391** | **0.824** | **0.406** | **56.5%** |

O recall de 82% em 2026 mostra que o limiar escolhido em 2025 se manteve (inclusive com os meses de jul–set, de maior movimento). O custo ainda é alto: para pegar 4 de cada 5 acidentes graves, o resgate avançado seria acionado em 57% das ligações (eram 64% antes das features de veículos e pessoas), e cerca de 2 em cada 5 acionamentos são de fato graves (contra 28% se todas as ligações fossem atendidas).

### Robustez ao relato incompleto

Para medir quanto do ganho das features de veículos e pessoas sobrevive quando o solicitante esquece ou não sabe o que há na pista, o teste 2026 foi degradado: com probabilidade *p*, a ligação inteira chega sem o bloco de veículos/pessoas. O modelo é treinado uma vez, com relato completo, e o limiar é o de 0.391.

| Ligações sem o bloco | PR-AUC | Recall | Acionamentos |
| :--- | :---: | :---: | :---: |
| 0% | 0.599 | 0.825 | 56.7% |
| 20% | 0.572 | 0.739 | 49.7% |
| 50% | 0.539 | 0.614 | 39.3% |
| 100% | 0.505 | 0.402 | 22.0% |

O ganho encolhe de forma quase linear. Com 50% de omissão a PR-AUC empata com a do modelo sem as features (0.539), e acima disso fica **pior** que ele: o modelo aprendeu que "0 motos, 0 pedestres" significa acidente leve e lê a omissão como ausência. O recall também cai, e a meta de 80% só vale com relato completo.

**A defesa foi treinar já com omissão simulada**: cada acidente do treino recebe uma taxa de omissão própria sorteada entre 0% e 100%, e o bloco omitido vira `NaN` ("não informado"), que o XGBoost trata nativamente. O limiar (0.389) é escolhido com a mesma validação temporal, mas com a validação também degradada.

| Ligações sem o bloco | PR-AUC | Recall | Precision | Acionamentos |
| :--- | :---: | :---: | :---: | :---: |
| 0% | 0.597 | 0.832 | 0.401 | 58.1% |
| 20% | 0.584 | 0.829 | 0.393 | 59.2% |
| 50% | 0.569 | 0.826 | 0.380 | 61.0% |
| 100% | 0.536 | 0.821 | 0.361 | 63.8% |

O recall se mantém em cerca de 82% em qualquer nível de omissão, e um limiar único serve (os limiares ótimos por nível ficaram entre 0.388 e 0.394). O preço é operacional: sem o relato de veículos, a precision cai de 0.40 para 0.36 e o acionamento sobe de 58% para 64%.

> **Ressalva:** a omissão é simulada, aleatória e igual no treino e no teste. Em ligações reais ela pode se correlacionar com a gravidade, e só relatos reais permitiriam medir isso.

### Calibração das probabilidades

Ranking e limiar só exigem que a ordem das ligações esteja certa; já mostrar "70% de chance de vítima grave" ao atendente exige que o número seja real. O XGBoost com `scale_pos_weight` **superestima**: no teste 2026 a probabilidade média prevista é 0.45 para uma taxa real de 0.28 (ECE 0.171; ligações que o modelo dá ~65% têm 42% de graves de fato). Um calibrador aprendido em 2025 (com omissão simulada, como no treino) corrige isso sem tocar no teste:

| Versão (relato completo) | Brier | ECE | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: |
| Bruta | 0.194 | 0.171 | 0.765 | 0.589 |
| Platt (sigmoide) | 0.162 | 0.008 | 0.765 | 0.589 |
| Isotônica | 0.162 | 0.007 | 0.765 | 0.584 |

A calibração é uma transformação monotônica, então o ranking e o limiar por recall não mudam. O Platt vale também em todos os níveis de omissão (ECE ≤ 0.008), sem ajustar por nível. Fica como recomendação o Platt (2 parâmetros, sem os empates da isotônica, que perde um pouco de PR-AUC).

#### Calibrador do modelo de produção

O calibrador acima foi ajustado sobre o modelo treinado só em 2024. O modelo de produção (2024 + 2025) não tem período de validação sobrando, então `src/calibration_prod.py` usa *cross-fitting*: previsões out-of-fold (5 folds) sobre 2024 + 2025, cada acidente previsto por um modelo que não o viu, e o Platt aprende sobre elas. No teste 2026 o ECE é 0.009 com relato completo e 0.005 sem nenhum relato (o Platt antigo dá 0.012 e 0.008), com Brier 0.160 a 0.172. Os dois são equivalentes na prática; o novo é o que acompanha o modelo salvo (`models/xgboost_nan_platt_prod.joblib`).

### Desempenho por subgrupo

O recall global de ~83% esconde diferenças grandes (`src/subgroups.py`, teste 2026, limiar único 0.389, IC 95% por bootstrap):

| Subgrupo | Graves | Recall | Acionamentos |
| :--- | :---: | :---: | :---: |
| Atropelamento de pedestre / colisão frontal | 68% / 63% | 1.00 / 1.00 | 99% / 99% |
| Pista simples | 33% | 0.90 | 70% |
| Pista dupla | 23% | 0.75 | 48% |
| Saída de leito carroçável | 22% | 0.59 | 38% |
| Engavetamento | 12% | 0.59 | 22% |
| DF (n = 804) | 19% | 0.51 | 28% |
| SP | 18% | 0.60 | 30% |

O padrão é a **prevalência**: onde graves são menos comuns, um limiar único de probabilidade pega menos deles (e aciona menos), e onde são comuns o modelo aciona quase sempre. Isso é coerente com um modelo calibrado e custo igual para todos os erros, mas significa que "recall de 80%" vale na média, não por tipo de acidente ou estado. Se a regra de negócio exigir recall mínimo por grupo (por exemplo, por UF), seria preciso limiares por grupo, ao custo de mais acionamentos nos grupos de baixa prevalência. Com 50% de omissão do relato o quadro é parecido (recall global 0.826; engavetamento cai para 0.54).

> Os testes de subgrupo são muitos (79), então alguns "significativamente abaixo" aparecem por acaso; os padrões acima (tipo de acidente, tipo de pista, SP/RJ/RS/DF) são grandes e consistentes o bastante para não serem só ruído.

#### Limiares por grupo

`src/group_threshold.py` escolhe um limiar *por grupo* (recall ≥ 80% dentro de cada tipo de acidente ou UF) na validação 2025, sem tocar no teste. Grupos com menos de 500 acidentes ou 100 graves na validação herdam o limiar global, porque o limiar de um grupo pequeno é só ruído. No teste 2026, com relato completo:

| Limiares | Recall | Precision | Acionamentos | Pior recall de um grupo | Grupos com recall < 70% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Único (0.389) | 0.831 | 0.401 | 57.9% | 0.51 (UF) / 0.59 (tipo) | 4 de 23 UFs / 4 de 13 tipos |
| Por UF | 0.834 | 0.389 | 59.7% | **0.76** | 0 |
| Por tipo de acidente | 0.821 | 0.374 | 61.2% | **0.75** | 0 |

Uniformizar a cobertura é **barato**: o pior recall sobe de ~0.5 para ~0.75 por UF ao custo de +1.8 ponto de acionamento. Faz sentido, porque os grupos descobertos são os de baixa prevalência, que têm poucas ligações; elevar o recall neles quase não pesa no total. Com 50% de omissão do relato o resultado é igual (pior recall ~0.75, +2 a +6 pontos de acionamento). Se a operação exigir cobertura uniforme, essa é a opção; se o objetivo é só o recall médio, o limiar único basta.

### Do limiar por recall ao limiar por custo

"Recall de 80%" é uma meta, não uma conta. Com a probabilidade calibrada, a decisão que minimiza o custo é acionar quando `p ≥ 1 / (1 + r)`, em que `r = C_fn / C_fp` é quanto custa perder um grave em relação a um acionamento desnecessário. `src/cost_analysis.py` mostra isso no teste 2026 (custo por ligação em unidades de `C_fp`):

| r | Limiar ótimo | Recall | Acionamentos | Custo (limiar ótimo) | Custo (recall 80%) | Nunca aciona | Sempre aciona |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2 | 0.333 | 0.59 | 31% | 0.377 | 0.441 | 0.558 | 0.721 |
| 5 | 0.167 | 0.89 | 67% | 0.576 | 0.583 | 1.395 | 0.721 |
| 10 | 0.091 | 0.98 | 88% | 0.665 | 0.818 | 2.789 | 0.721 |

(O limiar do recall 80%, 0.389 bruto, equivale a 0.202 calibrado, que é o ótimo para **r ≈ 4**.) Ou seja, a meta de 80% equivale a dizer que perder um grave custa cerca de 4 envios desnecessários. Se a PRF julgar que perder um grave é mais caro que isso, o limiar certo é menor e o recall maior (r = 10 pede recall 0.98, acionando 88% das ligações, com custo 0.67 contra 0.82 do limiar de recall 80%); se julgar menos, o limiar sobe. O modelo calibrado permite transformar essa decisão de negócio em limiar, em vez de escolher um recall por palpite. A calibração entra aqui: sem ela, `1 / (1 + r)` não valeria.

### Drift temporal

O modelo envelhece? `src/drift.py` faz uma avaliação rolante de jan/2025 a set/2026. Em cada mês compara um modelo **congelado** (treinado só em 2024) com um **retreinado** com todos os meses anteriores, no mesmo mês e com o limiar fixo de 0.389:

| | 1º semestre 2025 | Últimos 6 meses (abr–set 2026) |
| :--- | :---: | :---: |
| Ganho de PR-AUC do retreino sobre o congelado | +0.002 | +0.010 |

- **O envelhecimento existe, mas é lento e pequeno.** O ganho do retreino cresce com a idade do modelo (de +0.002 aos poucos meses para +0.010 com 2 anos), e o PR-AUC mensal varia ±0.03 por dificuldade do mês, bem mais que isso.
- **O recall se mantém perto de 80%** sem reajustar o limiar: o congelado oscila entre 0.79 e 0.83 (média 0.81) e o retreinado entre 0.80 e 0.83 (média 0.82). Cada mês tem ~1.7 mil graves, então ±0.01 já é ruído de amostragem.
- Em 2026 o modelo de produção (2024 + 2025) fica praticamente igual ao retreinado todo mês (PR-AUC 0.572 a 0.622 contra 0.576 a 0.625).
- **Conclusão prática:** retreinar uma vez por ano parece suficiente; vale monitorar o recall mensal e retreinar se ele ficar abaixo de ~0.78.

> Os meses de 2025 não são totalmente fora da amostra para o limiar, que foi escolhido em 2025. Os meses de 2026 são.

### Omissão não aleatória

A ressalva mais forte do projeto é que a omissão do relato foi simulada como aleatória. `src/informative_omission.py` testa o modelo de produção em 2026 com omissão que **depende da ocorrência**, comparando com omissão aleatória de mesma taxa média (limiar 0.389, média de 10 sorteios):

| Cenário (omissão ~50%) | Recall | Acionamentos | PR-AUC | Δ PR-AUC vs aleatória |
| :--- | :---: | :---: | :---: | :---: |
| Aleatória (referência) | 0.827 | 60.8% | 0.566 | – |
| Graves omitem mais (80% vs 38%) | 0.823 | 60.0% | 0.541 | -0.024 |
| Graves omitem menos (20% vs 62%) | 0.829 | 61.6% | 0.589 | +0.023 |
| Noite omite mais (80% vs 20%) | 0.833 | 61.3% | 0.564 | -0.005 |
| 3+ veículos omitem mais | 0.829 | 61.0% | 0.563 | -0.004 |
| **Adversário: moto ou pedestre omitem mais (80% vs 30%)** | **0.794** | **55.9%** | 0.562 | -0.001 |

- **O recall resiste quando a omissão depende da gravidade, da hora ou do nº de veículos** (0.82 a 0.83). Quando a omissão depende só do desfecho, o recall por classe não muda por construção; o que muda é a precision e a PR-AUC (±0.024): se os graves são os que mais omitem, o modelo fica mais cego, e o contrário o favorece.
- **O pior caso é a omissão ligada ao próprio conteúdo do bloco**: se o relato some justamente quando há moto ou pedestre, o recall cai de 0.83 para 0.79 (para 0.76 nos graves que informaram, porque o modelo "aprende" a ler o relato presente como sinal de caso leve). A queda é de 3 pontos, não um colapso, graças ao treino com `NaN`.
- Continua sendo simulação: as regras de omissão são inventadas. Só relatos reais de ligações diriam qual desses cenários se aproxima da realidade.

### Interpretabilidade (SHAP)

`src/explain.py` calcula valores SHAP exatos (TreeSHAP nativo do XGBoost) no teste 2026 e soma as contribuições de volta à variável original. As mais influentes são o que o atendente relata: `n_moto` (0.49 de |SHAP| médio, em log-odds), `tipo_acidente` (0.45), `pessoas` (0.21) e `n_pesado` (0.18), seguidas de `municipio` (0.15). Quando o bloco de veículos é omitido, `n_moto` e `n_pesado` caem para ~0.07 e ~0.01 e o modelo passa a depender de `tipo_acidente` e da localização, coerente com a robustez medida acima. Ter moto envolvida soma +0.46 de log-odds contra -0.50 sem moto; `Incêndio` puxa a previsão para baixo (-4.3 a -4.9 nos casos de menor probabilidade).

### Experimento: abordagem preventiva (só variáveis pré-evento)

Avaliado com a Regressão Logística:

| Modelo | ROC-AUC | PR-AUC |
| :--- | :---: | :---: |
| Triagem (contexto + relato do 191) | 0.757 | 0.551 |
| Preventivo (só contexto) | 0.633 | 0.401 |

Sem o relato da ocorrência (tipo de acidente, veículos, pessoas), o desempenho cai bastante. O contexto indica *onde* e *quando* o risco é maior, mas não separa bem os casos individuais. Detalhes no notebook.

> A acurácia do Dummy (72%) é maior que a do modelo porque a base é desbalanceada. Por isso as métricas principais são **ROC-AUC** e **PR-AUC**.

**Próximos passos:** avaliar uma classificação multiclasse (leve, grave, fatal); e validar a hipótese de omissão aleatória com relatos reais das ligações, se houver acesso a eles.

---

## 🧰 Como Executar o Projeto

### 1. Clonar o Repositório
```bash
git clone https://github.com/rodrigomsousa/prf-accident-severity-predictor.git
cd prf-accident-severity-predictor
```

### 2. Criar e Ativar Ambiente Virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 4. Baixar os Dados
Baixe, para os anos desejados, os arquivos **"Agrupados por ocorrência"** e **"Agrupados por pessoa"** no portal de [Dados Abertos da PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf) e salve-os em `data/raw/` como `datatran{ano}.csv` e `pessoas{ano}.csv`.

### 5. Treinar o Modelo
```bash
python src/train.py
python src/threshold.py   # escolhe o limiar de decisão do XGBoost
python src/threshold_nan.py   # limiar do XGBoost treinado com omissão
python src/robustness.py      # teste de robustez ao relato incompleto
python src/augmentation.py    # compara treinar com e sem omissão simulada
python src/calibration.py     # calibra as probabilidades do modelo aug-nan
python src/calibration_prod.py  # calibrador do modelo de produção
python src/subgroups.py       # recall por subgrupo
python src/group_threshold.py # limiares por grupo
python src/cost_analysis.py   # limiar por custo
python src/drift.py           # drift temporal, mês a mês (leva alguns minutos)
python src/informative_omission.py  # omissão não aleatória
python src/explain.py         # valores SHAP (gráfico em reports/)
```

### Demo e testes
```bash
streamlit run app/streamlit_app.py   # demo: descreva a ocorrência e veja probabilidade, decisão e SHAP
pytest                                # testes (os de modelo de produção são pulados sem data/ e models/)
```

A demo usa o modelo `aug-nan`, o Platt de produção e o limiar de recall 80%. Se o solicitante não informou os veículos, desmarque a caixa correspondente: o bloco vira `NaN` e a previsão usa só tipo de acidente e contexto.

### 6. Executar o Notebook
```bash
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

---

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
