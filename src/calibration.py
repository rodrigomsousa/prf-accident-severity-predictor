"""Calibração das probabilidades do XGBoost `aug-nan`.

Um modelo está calibrado quando, entre as ligações às quais ele dá ~30% de
chance, de fato ~30% são graves. Ranking (ROC-AUC/PR-AUC) e limiar só pedem que
a ordem esteja certa; calibração importa quando o número é mostrado ao
atendente ("70% de chance de vítima grave") ou entra em uma conta de custo.

O XGBoost aqui usa `scale_pos_weight` (reequilibra as classes no treino), o que
infla as probabilidades: espera-se que ele superestime a chance de grave.

Procedimento sem tocar no teste (igual ao `threshold_nan.py`): o modelo é
ajustado em 2024, o calibrador aprende em 2025 (com omissão simulada, como no
treino) e tudo é avaliado em 2026. Compara três versões:
- bruta:       probabilidade do XGBoost;
- Platt:       regressão logística sobre o logit da probabilidade bruta (2 parâmetros,
               corrige viés geral, robusta);
- isotônica:   função degrau monotônica (flexível, precisa de mais dados).

Calibrar é uma transformação monotônica: a ordem das ligações não muda, então
ROC-AUC e o limiar por recall seguem valendo (a isotônica pode criar empates).

Uso:
    python src/calibration.py
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from augmentation import LEVELS
from data_preprocessing import load_dataset
from degradation import augment_nan
from threshold import FIT_YEARS, VALID_YEARS
from threshold_nan import fit_aug_nan, proba_at
from train import MODELS_DIR, TEST_YEARS

N_BINS = 10
EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Platt:
    def fit(self, p: np.ndarray, y: pd.Series) -> "Platt":
        self.lr = LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(logit(p).reshape(-1, 1))[:, 1]


class Isotonic:
    def fit(self, p: np.ndarray, y: pd.Series) -> "Isotonic":
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self.iso.predict(p)


def ece(y: pd.Series, p: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected Calibration Error: média ponderada de |taxa real - prob. média|
    por faixa de probabilidade (faixas de largura igual)."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    y = np.asarray(y)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return total


def report(y: pd.Series, p: np.ndarray) -> dict[str, float]:
    return {"Brier": brier_score_loss(y, p), "ECE": ece(y, p),
            "ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p),
            "Prob. média": p.mean()}


def main() -> None:
    X_valid, y_valid = load_dataset(VALID_YEARS)
    X_test, y_test = load_dataset(TEST_YEARS)
    model = fit_aug_nan(FIT_YEARS, seed=0)

    # Validação com a mesma mistura de omissão do treino.
    p_valid = model.predict_proba(augment_nan(X_valid, np.random.default_rng(1)))[:, 1]
    calibrators = {"Platt": Platt().fit(p_valid, y_valid),
                   "Isotônica": Isotonic().fit(p_valid, y_valid)}

    print(f"Taxa real de graves: validação {y_valid.mean():.3f}, teste {y_test.mean():.3f}\n")
    rows, curves = [], {}
    for p in LEVELS:
        raw = proba_at(model, X_test, p, 1000)
        versions = {"bruta": raw, **{n: c.predict(raw) for n, c in calibrators.items()}}
        for name, proba in versions.items():
            rows.append({"omissão": p, "versão": name, **report(y_test, proba)})
            if p in (0.0, 1.0):
                curves[(p, name)] = calibration_curve(y_test, proba, n_bins=N_BINS)
    results = pd.DataFrame(rows).set_index(["omissão", "versão"])
    print(f"Métricas no teste {TEST_YEARS} (taxa real {y_test.mean():.3f}):")
    print(results.round(3).to_string(), "\n")

    print("Curva de confiabilidade (omissão 0%): prob. média prevista -> taxa real")
    for name in ["bruta", "Platt", "Isotônica"]:
        frac, mean = curves[(0.0, name)]
        print(f"  {name:10s}", " ".join(f"{m:.2f}->{f:.2f}" for m, f in zip(mean, frac)))

    # Calibrador final (Platt, mais estável) salvo para uso na inferência. É
    # ajustado no modelo de 2024; para o modelo de produção (2024+2025) seria
    # preciso reservar um período de validação à parte.
    joblib.dump(calibrators["Platt"], MODELS_DIR / "xgboost_nan_platt.joblib")
    (MODELS_DIR / "xgboost_nan_platt.json").write_text(json.dumps({
        "fit_on": FIT_YEARS, "calibrated_on": VALID_YEARS,
        "coef": float(calibrators["Platt"].lr.coef_[0, 0]),
        "intercept": float(calibrators["Platt"].lr.intercept_[0])}))


if __name__ == "__main__":
    main()
