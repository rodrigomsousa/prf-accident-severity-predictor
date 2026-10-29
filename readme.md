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
| Triagem: Logistic Regression | 0.706 | 0.482 | 0.651 | 0.554 | 0.757 | 0.551 |
| Triagem: Random Forest | 0.747 | 0.549 | 0.541 | 0.545 | 0.764 | 0.584 |
| **Triagem: XGBoost** | 0.710 | 0.487 | 0.673 | 0.565 | **0.774** | **0.599** |

Precision, recall e F1 desta tabela usam o limiar padrão de 0.5.

**O que mais melhorou o modelo foram as features de veículos e pessoas** (quantas motos, bicicletas, caminhões e ônibus, e se há pedestre), agregadas dos arquivos da PRF por pessoa. Sem elas, os três algoritmos ficavam colados em ROC-AUC ≈ 0.72. O XGBoost foi de 0.721 / 0.539 para 0.774 / 0.599 (ROC-AUC / PR-AUC), com intervalo de 95% do ganho em PR-AUC entre +0.056 e +0.065 (bootstrap pareado no teste). Com isso, as árvores passaram a se separar do modelo linear. Apenas o `estado_fisico` e demais colunas de desfecho dos arquivos por pessoa ficam de fora.

As features geográficas (coordenadas e histórico de município e trecho) trouxeram um ganho menor, mas estatisticamente real: +0.005 de PR-AUC, com intervalo de 95% entre +0.003 e +0.008.

> **Ressalva:** os arquivos por pessoa são o registro final da PRF. O modelo supõe que o atendente consegue saber na ligação se há moto ou pedestre, mas um relato incompleto ou errado reduziria o ganho em produção.

### Ajuste do limiar de decisão

O modelo devolve uma probabilidade, e o limiar decide a partir de qual valor o acidente recebe resgate avançado. Deixar de enviar resgate a um acidente grave custa mais caro do que um envio desnecessário, então o limiar foi ajustado para garantir **recall de pelo menos 80%**.

O limiar foi escolhido **sem olhar o teste**: o modelo treina em 2024, o limiar é escolhido em 2025 e só então o modelo é retreinado em 2024 + 2025 e avaliado em 2026.

| XGBoost no teste 2026 | Limiar | Recall | Precision | Acionamentos |
| :--- | :---: | :---: | :---: | :---: |
| Limiar padrão | 0.500 | 0.673 | 0.487 | 38.7% |
| **Limiar ajustado** | **0.391** | **0.825** | **0.408** | **56.7%** |

O recall de 82% em 2026 mostra que o limiar escolhido em 2025 se manteve. O custo ainda é alto: para pegar 4 de cada 5 acidentes graves, o resgate avançado seria acionado em 57% das ligações (eram 64% antes das features de veículos e pessoas), e cerca de 2 em cada 5 acionamentos são de fato graves (contra 28% se todas as ligações fossem atendidas).

### Experimento: abordagem preventiva (só variáveis pré-evento)

Avaliado com a Regressão Logística:

| Modelo | ROC-AUC | PR-AUC |
| :--- | :---: | :---: |
| Triagem (contexto + relato do 191) | 0.757 | 0.551 |
| Preventivo (só contexto) | 0.633 | 0.401 |

Sem o relato da ocorrência (tipo de acidente, veículos, pessoas), o desempenho cai bastante. O contexto indica *onde* e *quando* o risco é maior, mas não separa bem os casos individuais. Detalhes no notebook.

> A acurácia do Dummy (72%) é maior que a do modelo porque a base é desbalanceada. Por isso as métricas principais são **ROC-AUC** e **PR-AUC**.

**Próximos passos:** simular um relato incompleto ou errado (esconder parte das informações de veículos e pessoas no teste) para medir quanto do ganho sobrevive em produção; e avaliar uma classificação multiclasse (leve, grave, fatal).

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
```

### 6. Executar o Notebook
```bash
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

---

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
