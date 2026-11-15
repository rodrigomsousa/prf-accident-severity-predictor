"""Calibrador próprio para o modelo de produção (treinado em 2024 + 2025).

O `calibration.py` ajusta o Platt sobre o modelo de 2024 (calibra em 2025). O
modelo de produção, porém, é treinado em 2024 + 2025, e não sobra período de
validação "limpo" para calibrá-lo. A saída é o *cross-fitting*: previsões
out-of-fold (OOF). Cada acidente de 2024+2025 é previsto por um modelo que NÃO o
viu no treino (5 folds), e o Platt aprende sobre essas previsões honestas.

Cuidados:
- o treino é degradado uma única vez (cada acidente com sua taxa de omissão, bloco
  omitido = NaN), como no `aug-nan`; as previsões OOF usam a mesma mistura;
- os folds são aleatórios, não temporais. Para uma transformação de 2 parâmetros
  (Platt) isso basta, e o teste 2026 diz se funcionou.

Compara, no teste 2026 e em vários níveis de omissão, três versões da
probabilidade do modelo de produção: bruta, Platt antigo (ajustado no modelo de
2024) e Platt OOF (novo).

Uso:
    python src/calibration_prod.py
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from augmentation import LEVELS
from calibration import Platt, report
from data_preprocessing import TRIAGE, load_dataset
from degradation import augment_nan
from threshold import FIT_YEARS, VALID_YEARS
from threshold_nan import proba_at
from train import MODELS_DIR, TEST_YEARS, make_pipeline, make_xgboost

N_FOLDS = 5


def oof_proba(X: pd.DataFrame, y: pd.Series, seed: int = 0) -> np.ndarray:
    """Probabilidade out-of-fold do pipeline aug-nan em X (já degradado)."""
    oof = np.zeros(len(X))
    for train_idx, valid_idx in KFold(N_FOLDS, shuffle=True, random_state=seed).split(X):
        model = make_pipeline(TRIAGE, make_xgboost(y.iloc[train_idx]))
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
    return oof


def main() -> None:
    years = FIT_YEARS + VALID_YEARS
    X, y = load_dataset(years)
    X_aug = augment_nan(X, np.random.default_rng(0))
    platt_oof = Platt().fit(oof_proba(X_aug, y), y)
    print(f"Platt OOF ({years}): coef={platt_oof.lr.coef_[0, 0]:.3f} "
          f"intercept={platt_oof.lr.intercept_[0]:.3f}")

    final = joblib.load(MODELS_DIR / "xgboost_aug_nan.joblib")
    platt_old = joblib.load(MODELS_DIR / "xgboost_nan_platt.joblib")
    X_test, y_test = load_dataset(TEST_YEARS)

    rows = []
    for p in LEVELS:
        raw = proba_at(final, X_test, p, 1000)
        for name, proba in {"bruta": raw, "Platt antigo": platt_old.predict(raw),
                            "Platt OOF": platt_oof.predict(raw)}.items():
            rows.append({"omissão": p, "versão": name, **report(y_test, proba)})
    results = pd.DataFrame(rows).set_index(["omissão", "versão"])
    print(f"\nTeste {TEST_YEARS} (taxa real {y_test.mean():.3f}):")
    print(results.round(3).to_string())

    joblib.dump(platt_oof, MODELS_DIR / "xgboost_nan_platt_prod.joblib")
    (MODELS_DIR / "xgboost_nan_platt_prod.json").write_text(json.dumps({
        "model": "xgboost_aug_nan", "fit_on": years, "method": f"OOF {N_FOLDS}-fold",
        "coef": float(platt_oof.lr.coef_[0, 0]),
        "intercept": float(platt_oof.lr.intercept_[0])}))


if __name__ == "__main__":
    main()
