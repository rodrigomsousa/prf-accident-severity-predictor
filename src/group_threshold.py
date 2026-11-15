"""Limiares por grupo: cobertura uniforme de graves, a que custo?

Com um limiar único, o recall varia muito entre subgrupos (ver `subgroups.py`):
onde graves são raros (engavetamento, saída de leito, DF, SP) o modelo pega
menos deles. Aqui o limiar é escolhido *por grupo* (recall >= 80% dentro de cada
grupo) e comparado ao limiar único.

Sem tocar no teste: os limiares são escolhidos na validação 2025 (modelo
ajustado em 2024, validação degradada como no treino) e avaliados em 2026 com o
modelo de produção. Grupos pequenos ou com poucos graves na validação herdam o
limiar global, porque o limiar de um grupo com poucos casos é só ruído.

Uso:
    python src/group_threshold.py
"""

import joblib
import numpy as np
import pandas as pd

from data_preprocessing import load_dataset
from degradation import augment_nan
from subgroups import add_groups
from threshold import FIT_YEARS, TARGET_RECALL, VALID_YEARS, threshold_for_recall
from threshold_nan import fit_aug_nan, proba_at
from train import MODELS_DIR, TEST_YEARS

MIN_N = 500       # acidentes do grupo na validação
MIN_GRAVES = 100  # graves do grupo na validação
GROUP_COLS = ["tipo_acidente", "uf"]


def group_thresholds(G: pd.DataFrame, y: pd.Series, proba: np.ndarray, col: str,
                     default: float) -> dict[str, float]:
    thr = {}
    for value, idx in G.groupby(col).indices.items():
        yy = y.to_numpy()[idx]
        thr[value] = (threshold_for_recall(yy, proba[idx], TARGET_RECALL)
                      if len(idx) >= MIN_N and yy.sum() >= MIN_GRAVES else default)
    return thr


def evaluate(G: pd.DataFrame, y: np.ndarray, proba: np.ndarray, thr_per_row: np.ndarray,
             col: str) -> dict[str, float]:
    pred = proba >= thr_per_row
    rec = {v: pred[idx][y[idx] == 1].mean() for v, idx in G.groupby(col).indices.items()
           if y[idx].sum() >= 30 and len(idx) >= 300}
    rec = pd.Series(rec)
    return {"Recall": pred[y == 1].mean(), "Precision": y[pred].mean(),
            "Acionamento": pred.mean(), "Recall mín. (grupo)": rec.min(),
            "Grupos com recall < 70%": int((rec < 0.70).sum()), "Grupos": len(rec)}


def main() -> None:
    X_valid, y_valid = load_dataset(VALID_YEARS)
    fit_model = fit_aug_nan(FIT_YEARS, seed=0)
    p_valid = fit_model.predict_proba(augment_nan(X_valid, np.random.default_rng(1)))[:, 1]
    global_thr = threshold_for_recall(y_valid, p_valid, TARGET_RECALL)
    G_valid = add_groups(X_valid)

    prod = joblib.load(MODELS_DIR / "xgboost_aug_nan.joblib")
    X_test, y_test = load_dataset(TEST_YEARS)
    G_test = add_groups(X_test)
    yt = y_test.to_numpy()

    for p_omit in [0.0, 0.5]:
        proba = proba_at(prod, X_test, p_omit, 7)
        print(f"\n=== Teste {TEST_YEARS}, omissão {p_omit:.0%} (limiar global {global_thr:.3f}) ===")
        for col in GROUP_COLS:
            thr = group_thresholds(G_valid, y_valid, p_valid, col, global_thr)
            n_own = sum(t != global_thr for t in thr.values())
            per_row = G_test[col].map(thr).fillna(global_thr).to_numpy()
            rows = {"Limiar único": evaluate(G_test, yt, proba, np.full(len(yt), global_thr), col),
                    f"Por {col}": evaluate(G_test, yt, proba, per_row, col)}
            print(f"\nPor {col} ({n_own} de {len(thr)} grupos com limiar próprio)")
            print(pd.DataFrame(rows).T.round(3).to_string())


if __name__ == "__main__":
    main()
