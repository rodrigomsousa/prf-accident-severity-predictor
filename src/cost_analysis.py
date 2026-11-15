"""Do limiar por recall ao limiar por custo.

Escolher "recall >= 80%" é uma meta, não uma conta. A decisão ótima de despacho
vem de comparar custos. Seja:
- C_fn: custo de NÃO enviar resgate avançado a um acidente grave;
- C_fp: custo de enviar sem necessidade.

Com probabilidade calibrada p, enviar custa (1 - p) * C_fp e não enviar custa
p * C_fn. Envia-se quando p >= C_fp / (C_fp + C_fn) = 1 / (1 + r), com
r = C_fn / C_fp. Por isso a calibração (seção 4.6) importa: o limiar ótimo só
vale se p for uma probabilidade de verdade.

Este script (1) mede o custo no teste 2026 para vários r, comparando o limiar
ótimo de cada r com "nunca enviar", "sempre enviar" e o limiar de recall 80%;
(2) calcula o r *implícito* no limiar atual: o quanto a PRF estaria dizendo que
perder um grave custa mais que um envio desnecessário.

O custo é medido em unidades de C_fp por ligação: r * FN/N + FP/N.

Uso:
    python src/cost_analysis.py
"""

import json

import joblib
import numpy as np
import pandas as pd

from data_preprocessing import load_dataset
from threshold_nan import proba_at
from train import MODELS_DIR, TEST_YEARS

RATIOS = [1, 2, 3, 5, 8, 10, 15, 20]


def cost(y: np.ndarray, pred: np.ndarray, r: float) -> float:
    """Custo médio por ligação, em unidades de C_fp."""
    fn = ((pred == 0) & (y == 1)).mean()
    fp = ((pred == 1) & (y == 0)).mean()
    return r * fn + fp


def main() -> None:
    model = joblib.load(MODELS_DIR / "xgboost_aug_nan.joblib")
    platt = joblib.load(MODELS_DIR / "xgboost_nan_platt_prod.joblib")
    thr_raw = json.loads((MODELS_DIR / "xgboost_nan_threshold.json").read_text())["threshold"]
    X, y = load_dataset(TEST_YEARS)
    yv = y.to_numpy()
    p = platt.predict(proba_at(model, X, 0.0, 7))  # probabilidade calibrada
    thr_cal = float(platt.predict(np.array([thr_raw]))[0])
    r_implicit = (1 - thr_cal) / thr_cal
    print(f"Limiar de recall 80%: bruto {thr_raw:.3f} = {thr_cal:.3f} calibrado "
          f"-> custo implícito r = C_fn/C_fp = {r_implicit:.2f}\n"
          "(perder um grave 'custa' isso vezes um envio desnecessário)\n")

    rows = []
    for r in RATIOS:
        t = 1 / (1 + r)
        pred = (p >= t).astype(int)
        rows.append({
            "r = C_fn/C_fp": r, "Limiar ótimo": t, "Recall": pred[yv == 1].mean(),
            "Precision": yv[pred == 1].mean(), "Acionamento": pred.mean(),
            "Custo (limiar ótimo)": cost(yv, pred, r),
            "Custo (recall 80%)": cost(yv, (p >= thr_cal).astype(int), r),
            "Custo (nunca envia)": cost(yv, np.zeros_like(yv), r),
            "Custo (sempre envia)": cost(yv, np.ones_like(yv), r),
        })
    print(f"Teste {TEST_YEARS}, custo médio por ligação (unidades de C_fp):")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
