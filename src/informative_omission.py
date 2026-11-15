"""Omissão não aleatória: e se esquecer o relato depender da ocorrência?

O modelo `aug-nan` foi treinado e testado com omissão *aleatória* e igual para
todos (cada ligação perde o bloco de veículos/pessoas com taxa p ~ U(0,1)). Em
ligações reais é provável que a omissão se correlacione com a gravidade ou com o
tipo de ocorrência: quem liga de um acidente grave pode estar em pânico e dar
menos detalhes; de madrugada há menos testemunhas; com muitos veículos o relato
fica confuso.

Aqui o modelo de produção é testado em 2026 sob omissão que depende de algo, com
o limiar fixo (0.389), e comparado à omissão aleatória *com a mesma taxa média*,
para isolar o efeito de a omissão ser informativa e não o de faltar informação.

Cenários (todos com taxa média parecida, ~40-50%):
- aleatória:            todas as ligações com a mesma taxa (referência);
- graves omitem mais:   graves 80%, leves ajustado para a média ser 50%;
- graves omitem menos:  graves 20%, leves ajustado para a média ser 50%;
- noite omite mais:     18h-5h com 80%, resto 20%;
- muitos veículos:      3+ veículos com 80%, resto 40%;
- moto/pedestre omite:  cenário adversário: se há moto ou pedestre (o sinal mais
                        forte do modelo) a omissão é 80%, senão 30%.

Limite: continua sendo simulação (a regra de omissão é inventada); mostra a
sensibilidade do modelo, não a omissão real.

Uso:
    python src/informative_omission.py
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from data_preprocessing import load_dataset
from degradation import hide_block_nan
from train import MODELS_DIR, TEST_YEARS

N_REPEATS = 10
TARGET_MEAN = 0.5


def scenarios(X: pd.DataFrame, y: np.ndarray) -> dict[str, np.ndarray]:
    """Probabilidade de omissão por ligação em cada cenário."""
    n, prev = len(X), y.mean()

    def graves(p_grave: float) -> np.ndarray:
        p_leve = (TARGET_MEAN - prev * p_grave) / (1 - prev)  # mantém a média em 50%
        return np.where(y == 1, p_grave, p_leve)

    hora = X["hora"].astype(int).to_numpy()
    return {
        "aleatória": np.full(n, TARGET_MEAN),
        "graves omitem mais": graves(0.8),
        "graves omitem menos": graves(0.2),
        "noite omite mais": np.where((hora >= 18) | (hora <= 5), 0.8, 0.2),
        "3+ veículos omitem mais": np.where(X["veiculos"].to_numpy() >= 3, 0.8, 0.4),
        "moto/pedestre omite (adversário)": np.where(
            (X["n_moto"].to_numpy() > 0) | (X["tem_pedestre"].to_numpy() > 0), 0.8, 0.3),
    }


def run(model, X: pd.DataFrame, y: np.ndarray, p_omit: np.ndarray, thr: float,
        seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    hidden = rng.random(len(X)) < p_omit
    proba = model.predict_proba(hide_block_nan(X, hidden))[:, 1]
    pred = proba >= thr
    return {"Omissão média": hidden.mean(), "Recall": pred[y == 1].mean(),
            "Precision": y[pred].mean(), "Acionamento": pred.mean(),
            "ROC-AUC": roc_auc_score(y, proba), "PR-AUC": average_precision_score(y, proba),
            "Recall graves omitidos": pred[(y == 1) & hidden].mean(),
            "Recall graves informados": pred[(y == 1) & ~hidden].mean(),
            "% graves omitidos": hidden[y == 1].mean()}


def main() -> None:
    model = joblib.load(MODELS_DIR / "xgboost_aug_nan.joblib")
    thr = json.loads((MODELS_DIR / "xgboost_nan_threshold.json").read_text())["threshold"]
    X, y_series = load_dataset(TEST_YEARS)
    y = y_series.to_numpy()

    rows = {}
    for name, p_omit in scenarios(X, y).items():
        reps = pd.DataFrame([run(model, X, y, p_omit, thr, s) for s in range(N_REPEATS)])
        rows[name] = reps.mean()
    res = pd.DataFrame(rows).T

    # Referência: omissão aleatória com a mesma taxa média de cada cenário.
    ref = {}
    for name in res.index:
        p = res.loc[name, "Omissão média"]
        reps = pd.DataFrame([run(model, X, y, np.full(len(X), p), thr, 100 + s)
                             for s in range(N_REPEATS)])
        ref[name] = reps.mean()
    ref = pd.DataFrame(ref).T

    pd.set_option("display.width", 200)
    print(f"Teste {TEST_YEARS}, limiar {thr:.3f} (média de {N_REPEATS} sorteios)\n")
    cols = ["Omissão média", "% graves omitidos", "Recall", "Precision", "Acionamento", "PR-AUC"]
    print(res[cols].round(3).to_string(), "\n")
    delta = (res[["Recall", "Precision", "Acionamento", "PR-AUC"]]
             - ref[["Recall", "Precision", "Acionamento", "PR-AUC"]])
    print("Diferença para omissão aleatória com a mesma taxa média:")
    print(delta.round(3).to_string(), "\n")
    print("Recall dos graves, por terem sido omitidos ou não:")
    print(res[["Recall graves omitidos", "Recall graves informados"]].round(3).to_string())


if __name__ == "__main__":
    main()
