"""Ajuste do limiar de decisão do XGBoost priorizando recall.

O modelo devolve uma probabilidade; o limiar decide a partir de qual valor
o acidente é tratado como grave (e recebe resgate avançado). O padrão 0.5 é
arbitrário: aqui escolhemos o limiar que garante um recall mínimo.

O limiar é escolhido sem tocar no teste, numa validação temporal dentro do
treino (ajusta em 2024, escolhe em 2025). Só então o modelo é retreinado em
2024 + 2025 e avaliado em 2026 com o limiar já fixado.

Uso:
    python src/threshold.py
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, precision_score, recall_score

from data_preprocessing import TRIAGE, load_dataset
from train import MODELS_DIR, TEST_YEARS, make_pipeline, make_xgboost

FIT_YEARS = [2024]
VALID_YEARS = [2025]
TARGET_RECALL = 0.80


def threshold_for_recall(y: pd.Series, y_proba: np.ndarray, target: float) -> float:
    """Maior limiar cujo recall ainda atinge `target`. Quanto maior o limiar,
    menos alarmes falsos, então queremos o maior que cumpra a meta."""
    _, recall, thresholds = precision_recall_curve(y, y_proba)
    # recall tem um elemento a mais que thresholds (o ponto final recall=0).
    return float(thresholds[recall[:-1] >= target].max())


def metrics_at(y: pd.Series, y_proba: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "Limiar": threshold,
        "Recall": recall_score(y, y_pred),
        "Precision": precision_score(y, y_pred, zero_division=0),
        # Fração das ligações que acionariam resgate avançado: o custo operacional.
        "Taxa de acionamento": y_pred.mean(),
    }


def fit_and_predict(train_years: list[int], eval_years: list[int]) -> tuple[pd.Series, np.ndarray]:
    X_train, y_train = load_dataset(train_years)
    X_eval, y_eval = load_dataset(eval_years)
    model = make_pipeline(TRIAGE, make_xgboost(y_train)).fit(X_train, y_train)
    return y_eval, model.predict_proba(X_eval)[:, 1]


def main() -> None:
    y_valid, proba_valid = fit_and_predict(FIT_YEARS, VALID_YEARS)
    threshold = threshold_for_recall(y_valid, proba_valid, TARGET_RECALL)
    print(f"Limiar escolhido na validação {VALID_YEARS} (recall >= {TARGET_RECALL:.0%}): "
          f"{threshold:.3f}\n")

    y_test, proba_test = fit_and_predict(FIT_YEARS + VALID_YEARS, TEST_YEARS)
    results = {
        "Padrão (0.5)": metrics_at(y_test, proba_test, 0.5),
        f"Ajustado (recall {TARGET_RECALL:.0%})": metrics_at(y_test, proba_test, threshold),
    }
    print(f"XGBoost no teste {TEST_YEARS} ({y_test.mean():.1%} graves):")
    print(pd.DataFrame(results).T.round(3).to_string())

    # Curva completa no teste, só para leitura do trade-off (não para escolher).
    print("\nTrade-off no teste por limiar:")
    grid = [metrics_at(y_test, proba_test, t) for t in np.arange(0.2, 0.75, 0.05)]
    print(pd.DataFrame(grid).round(3).to_string(index=False))

    MODELS_DIR.mkdir(exist_ok=True)
    (MODELS_DIR / "xgboost_threshold.json").write_text(json.dumps(
        {"threshold": threshold, "target_recall": TARGET_RECALL, "chosen_on": VALID_YEARS}))


if __name__ == "__main__":
    main()
