"""Desempenho por subgrupo: o recall global esconde onde o modelo falha?

O limiar foi escolhido para um recall global de ~80%. Em triagem de resgate, o
que importa é que nenhum tipo de acidente ou região fique sistematicamente
descoberto. Aqui o modelo de produção (`aug-nan`, limiar fixo) é avaliado no
teste 2026 separado por UF, tipo de acidente, tipo de pista, fase do dia, faixa
horária, clima, mês e rodovia.

Para cada subgrupo (com ao menos MIN_N acidentes): prevalência de graves, recall,
precision, taxa de acionamento e ROC-AUC. O recall vem com IC 95% de bootstrap,
porque um subgrupo pequeno oscila só por acaso; sinalizamos os que ficam
*significativamente* abaixo do recall global (limite superior do IC < global).

Uso:
    python src/subgroups.py [omissao]     # omissão p do relato (padrão 0.0 = completo)
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data_preprocessing import load_dataset
from threshold_nan import proba_at
from train import MODELS_DIR, TEST_YEARS

MIN_N = 300
N_BOOT = 500
GROUPS = ["uf", "tipo_acidente", "tipo_pista", "fase_dia", "faixa_hora",
          "condicao_metereologica", "mes", "br_top"]
OUT = Path(__file__).resolve().parents[1] / "data" / "subgroup_results.csv"


def recall_ci(y: np.ndarray, pred: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """IC 95% (percentil) do recall por bootstrap só entre os graves."""
    hits = pred[y == 1]
    if len(hits) == 0:
        return np.nan, np.nan
    boot = rng.choice(hits, size=(N_BOOT, len(hits))).mean(axis=1)
    return tuple(np.percentile(boot, [2.5, 97.5]))


def add_groups(X: pd.DataFrame) -> pd.DataFrame:
    g = X.copy()
    hora = g["hora"].astype(int)
    g["faixa_hora"] = pd.cut(hora, [-1, 5, 11, 17, 23],
                             labels=["00-05h", "06-11h", "12-17h", "18-23h"]).astype(str)
    top = g["br"].value_counts().index[:15]
    g["br_top"] = g["br"].where(g["br"].isin(top), "outras")
    return g


def main() -> None:
    p_omit = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    model = joblib.load(MODELS_DIR / "xgboost_aug_nan.joblib")
    thr = json.loads((MODELS_DIR / "xgboost_nan_threshold.json").read_text())["threshold"]
    X, y = load_dataset(TEST_YEARS)
    proba = proba_at(model, X, p_omit, 7)
    pred = (proba >= thr).astype(int)
    yv = y.to_numpy()
    rng = np.random.default_rng(0)

    overall_recall = pred[yv == 1].mean()
    print(f"Teste {TEST_YEARS}, omissão {p_omit:.0%}, limiar {thr:.3f}: "
          f"recall {overall_recall:.3f}, precision {yv[pred == 1].mean():.3f}, "
          f"acionamento {pred.mean():.1%}, ROC-AUC {roc_auc_score(yv, proba):.3f}\n")

    G = add_groups(X)
    rows = []
    for col in GROUPS:
        for value, idx in G.groupby(col).indices.items():
            if len(idx) < MIN_N or yv[idx].sum() < 30:
                continue
            yy, pp = yv[idx], pred[idx]
            lo, hi = recall_ci(yy, pp, rng)
            rows.append({
                "grupo": col, "valor": value, "n": len(idx), "graves": yy.mean(),
                "recall": pp[yy == 1].mean(), "ic_lo": lo, "ic_hi": hi,
                "precision": yy[pp == 1].mean() if pp.sum() else np.nan,
                "acionamento": pp.mean(),
                "roc_auc": roc_auc_score(yy, proba[idx]) if 0 < yy.sum() < len(yy) else np.nan,
                "abaixo_global": hi < overall_recall,
            })
    res = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    res.to_csv(OUT, index=False)

    pd.set_option("display.width", 200)
    cols = ["grupo", "valor", "n", "graves", "recall", "ic_lo", "ic_hi", "precision", "acionamento"]
    flagged = res[res["abaixo_global"]].sort_values("recall")
    print(f"Subgrupos com recall significativamente abaixo do global ({len(flagged)} de {len(res)}):")
    print(flagged[cols].round(3).to_string(index=False))
    for col in ["tipo_acidente", "uf", "faixa_hora", "tipo_pista"]:
        print(f"\n{col} (ordenado por recall):")
        print(res[res["grupo"] == col].sort_values("recall")[cols[1:]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
