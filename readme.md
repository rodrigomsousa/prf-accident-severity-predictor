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
* **Relato do solicitante:** tipo de acidente, número de veículos e de pessoas envolvidas.

Como experimento complementar, o notebook avalia uma **abordagem preventiva**: prever a gravidade só com variáveis pré-evento, para apoiar o planejamento do patrulhamento.

---

## 📊 Dataset & Dados Abertos

Dados do portal de **Dados Abertos da PRF** (agrupados por ocorrência): 2024, 2025 e 2026 (até agosto), cerca de 194 mil acidentes.

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
├── data/raw/              # CSVs da PRF: datatran{ano}.csv (ignorado no git)
├── models/                # Pipelines treinados (.joblib)
├── notebooks/
│   └── 01_eda_and_modeling.ipynb   # EDA, modelagem e experimento preventivo
├── src/
│   ├── data_preprocessing.py       # Carga, alvo, features e conjuntos de features
│   ├── train.py                    # Pipeline, treino, avaliação e persistência
│   └── threshold.py                # Ajuste do limiar de decisão priorizando recall
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
| Triagem: Logistic Regression | 0.687 | 0.454 | 0.580 | 0.510 | 0.715 | 0.524 |
| Triagem: Random Forest | 0.728 | 0.515 | 0.473 | 0.493 | 0.717 | 0.530 |
| **Triagem: XGBoost** | 0.686 | 0.454 | 0.592 | 0.514 | **0.721** | **0.539** |

Os três algoritmos ficam muito próximos (ROC-AUC entre 0.71 e 0.72), o que indica que o limite está nas features e não no algoritmo. Precision, recall e F1 desta tabela usam o limiar padrão de 0.5.

As features geográficas (coordenadas e histórico de município e trecho) trouxeram um ganho pequeno, mas estatisticamente real: o XGBoost foi de 0.718 / 0.534 para 0.721 / 0.539 (ROC-AUC / PR-AUC), com intervalo de 95% do ganho em PR-AUC entre +0.003 e +0.007 (bootstrap pareado no teste).

### Ajuste do limiar de decisão

O modelo devolve uma probabilidade, e o limiar decide a partir de qual valor o acidente recebe resgate avançado. Deixar de enviar resgate a um acidente grave custa mais caro do que um envio desnecessário, então o limiar foi ajustado para garantir **recall de pelo menos 80%**.

O limiar foi escolhido **sem olhar o teste**: o modelo treina em 2024, o limiar é escolhido em 2025 e só então o modelo é retreinado em 2024 + 2025 e avaliado em 2026.

| XGBoost no teste 2026 | Limiar | Recall | Precision | Acionamentos |
| :--- | :---: | :---: | :---: | :---: |
| Limiar padrão | 0.500 | 0.592 | 0.454 | 36.5% |
| **Limiar ajustado** | **0.389** | **0.825** | **0.360** | **64.2%** |

O recall de 82% em 2026 mostra que o limiar escolhido em 2025 se manteve. O custo é alto: para pegar 4 de cada 5 acidentes graves, o resgate avançado seria acionado em 64% das ligações, e só cerca de 1 em cada 3 acionamentos é de fato grave (contra 28% se todas as ligações fossem atendidas). Isso confirma que faltam ao modelo informações que separem melhor os casos.

### Experimento: abordagem preventiva (só variáveis pré-evento)

Avaliado com a Regressão Logística:

| Modelo | ROC-AUC | PR-AUC |
| :--- | :---: | :---: |
| Triagem (contexto + relato do 191) | 0.715 | 0.524 |
| Preventivo (só contexto) | 0.633 | 0.401 |

Sem o relato da ocorrência (tipo de acidente, veículos, pessoas), o desempenho cai bastante. O contexto indica *onde* e *quando* o risco é maior, mas não separa bem os casos individuais. Detalhes no notebook.

> A acurácia do Dummy (72%) é maior que a do modelo porque a base é desbalanceada. Por isso as métricas principais são **ROC-AUC** e **PR-AUC**.

**Próximos passos:** features sobre os veículos e pessoas envolvidos (ex.: motocicleta, pedestre), a partir dos dados da PRF por pessoa e por veículo, como aproximação do que o atendente do 191 pode perguntar na ligação.

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
Baixe os arquivos **"Agrupados por ocorrência"** dos anos desejados no portal de [Dados Abertos da PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf) e salve-os em `data/raw/` como `datatran{ano}.csv`.

### 5. Treinar o Modelo
```bash
python src/train.py
python src/threshold.py   # escolhe o limiar de decisão do XGBoost
```

### 6. Executar o Notebook
```bash
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

---

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
