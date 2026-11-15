"""Drift temporal: o modelo envelhece? Com que frequência convém retreinar?

Avaliação rolante, mês a mês, de jan/2025 a set/2026. Para cada mês de teste
compara dois modelos `aug-nan` (omissão simulada no treino, como em produção):
- congelado:  treinado só em 2024 e nunca atualizado;
- retreinado: treinado com *todos* os meses anteriores ao mês avaliado
              (janela crescente), como se retreinasse todo mês.

Os dois são avaliados no mesmo mês, então diferenças de dificuldade entre meses
(chuva, feriados, prevalência) afetam os dois igualmente; o que sobra de
diferença é o efeito de envelhecer. O limiar é fixo (0.389, o de produção) em
ambos: mede-se o quanto o recall se mantém sem reajustar.

Para os meses de 2026 também entra o modelo de produção salvo (2024+2025), que
é o que na prática estaria em uso.

Uso:
    python src/drift.py
"""

from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from data_preprocessing import TRIAGE, build_features, build_target, load_raw
from degradation import augment_nan
from threshold_nan import proba_at
from train import MODELS_DIR, make_pipeline, make_xgboost

OUT = Path(__file__).resolve().parents[1] / "data" / "drift_results.csv"
FIRST_TEST = pd.Period("2025-01", "M")
OMISSIONS = [0.0, 0.5]


def fit_window(X: pd.DataFrame, y: pd.Series, seed: int = 0):
    X_aug = augment_nan(X, np.random.default_rng(seed))
    return make_pipeline(TRIAGE, make_xgboost(y)).fit(X_aug, y)


def month_metrics(y: np.ndarray, proba: np.ndarray, thr: float) -> dict[str, float]:
    pred = proba >= thr
    return {"ROC-AUC": roc_auc_score(y, proba), "PR-AUC": average_precision_score(y, proba),
            "Recall": pred[y == 1].mean(), "Precision": y[pred].mean() if pred.any() else np.nan,
            "Acionamento": pred.mean(), "Prob. média": proba.mean()}


def main() -> None:
    raw = load_raw([2024, 2025, 2026])
    X, y = build_features(raw), build_target(raw)
    month = pd.to_datetime(raw["data_inversa"]).dt.to_period("M")
    thr = json.loads((MODELS_DIR / "xgboost_nan_threshold.json").read_text())["threshold"]
    prod = joblib.load(MODELS_DIR / "xgboost_aug_nan.joblib")

    frozen = fit_window(X[month < FIRST_TEST], y[month < FIRST_TEST])
    rows = []
    for m in sorted(month[month >= FIRST_TEST].unique()):
        test = (month == m).to_numpy()
        yt = y[test].to_numpy()
        retrained = fit_window(X[month < m], y[month < m])
        models = {"congelado (2024)": frozen, "retreinado": retrained}
        if m.year == 2026:
            models["produção (2024+2025)"] = prod
        for name, model in models.items():
            for p in OMISSIONS:
                proba = proba_at(model, X[test], p, 7)
                rows.append({"mês": str(m), "modelo": name, "omissão": p, "n": test.sum(),
                             "graves": yt.mean(), **month_metrics(yt, proba, thr)})
        print(f"{m} ok", flush=True)

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    res.to_csv(OUT, index=False)

    pd.set_option("display.width", 200)
    for p in OMISSIONS:
        sub = res[res["omissão"] == p]
        print(f"\n=== omissão {p:.0%}: PR-AUC e recall por mês (limiar fixo {thr:.3f}) ===")
        wide = sub.pivot(index="mês", columns="modelo", values=["PR-AUC", "Recall"])
        wide.insert(0, "graves", sub.drop_duplicates("mês").set_index("mês")["graves"])
        print(wide.round(3).to_string())
        paired = sub.pivot(index="mês", columns="modelo", values="PR-AUC")
        gap = (paired["retreinado"] - paired["congelado (2024)"])
        print(f"\nGanho médio de PR-AUC do retreino: {gap.mean():+.4f} "
              f"(1º semestre de 2025: {gap.iloc[:6].mean():+.4f}; "
              f"últimos 6 meses: {gap.iloc[-6:].mean():+.4f})")
        rc = sub.pivot(index="mês", columns="modelo", values="Recall")
        print(f"Recall médio: congelado {rc['congelado (2024)'].mean():.3f} "
              f"(mín. {rc['congelado (2024)'].min():.3f}); "
              f"retreinado {rc['retreinado'].mean():.3f} (mín. {rc['retreinado'].min():.3f})")


if __name__ == "__main__":
    main()
