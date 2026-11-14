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

Dados do portal de **Dados Abertos da PRF** (agrupados por ocorrência e por pessoa): 2024, 2025 e 2026 (até agosto), cerca de 194 mil acidentes e 524 mil pessoas envolvidas.

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
│   └── explain.py                  # Interpretabilidade com valores SHAP
├── .gitignore
├── readme.md
└── requirements.txt
```

---

## 📈 Resultados e Desempenho

Teste em **2026** (48.712 acidentes, 28% graves):

| Modelo | Acurácia | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (Dummy, sempre "leve") | 0.720 | 0.000 | 0.000 | 0.000 | 0.500 | 0.280 |
| Triagem: Logistic Regression | 0.706 | 0.482 | 0.651 | 0.554 | 0.757 | 0.551 |
| Triagem: Random Forest | 0.747 | 0.549 | 0.541 | 0.545 | 0.764 | 0.584 |
| **Triagem: XGBoost** | 0.710 | 0.487 | 0.673 | 0.565 | **0.774** | **0.599** |
| Triagem: XGBoost treinado com omissão (relato completo no teste) | 0.707 | 0.484 | 0.671 | 0.562 | 0.770 | 0.597 |

Precision, recall e F1 desta tabela usam o limiar padrão de 0.5. A última linha é o modelo preparado para relatos incompletos (veja a seção de robustez abaixo): com relato completo, perde só 0.002 de PR-AUC.

**O que mais melhorou o modelo foram as features de veículos e pessoas** (quantas motos, bicicletas, caminhões e ônibus, e se há pedestre), agregadas dos arquivos da PRF por pessoa. Sem elas, os três algoritmos ficavam colados em ROC-AUC ≈ 0.72. O XGBoost foi de 0.721 / 0.539 para 0.774 / 0.599 (ROC-AUC / PR-AUC), com intervalo de 95% do ganho em PR-AUC entre +0.056 e +0.065 (bootstrap pareado no teste). Com isso, as árvores passaram a se separar do modelo linear. Apenas o `estado_fisico` e demais colunas de desfecho dos arquivos por pessoa ficam de fora.

As features geográficas (coordenadas e histórico de município e trecho) trouxeram um ganho menor, mas estatisticamente real: +0.005 de PR-AUC, com intervalo de 95% entre +0.003 e +0.008.

> **Ressalva:** os arquivos por pessoa são o registro final da PRF. O modelo supõe que o atendente consegue saber na ligação se há moto ou pedestre. A seção de robustez abaixo mede o que acontece quando o relato é incompleto.

### Ajuste do limiar de decisão

O modelo devolve uma probabilidade, e o limiar decide a partir de qual valor o acidente recebe resgate avançado. Deixar de enviar resgate a um acidente grave custa mais caro do que um envio desnecessário, então o limiar foi ajustado para garantir **recall de pelo menos 80%**.

O limiar foi escolhido **sem olhar o teste**: o modelo treina em 2024, o limiar é escolhido em 2025 e só então o modelo é retreinado em 2024 + 2025 e avaliado em 2026.

| XGBoost no teste 2026 | Limiar | Recall | Precision | Acionamentos |
| :--- | :---: | :---: | :---: | :---: |
| Limiar padrão | 0.500 | 0.673 | 0.487 | 38.7% |
| **Limiar ajustado** | **0.391** | **0.825** | **0.408** | **56.7%** |

O recall de 82% em 2026 mostra que o limiar escolhido em 2025 se manteve. O custo ainda é alto: para pegar 4 de cada 5 acidentes graves, o resgate avançado seria acionado em 57% das ligações (eram 64% antes das features de veículos e pessoas), e cerca de 2 em cada 5 acionamentos são de fato graves (contra 28% se todas as ligações fossem atendidas).

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
python src/explain.py         # valores SHAP (gráfico em reports/)
```

### 6. Executar o Notebook
```bash
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

---

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
