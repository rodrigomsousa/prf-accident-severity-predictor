"""Ajuste do limiar do XGBoost treinado com omissão (`aug-nan`).

Mesmo procedimento do `threshold.py` (ajusta em 2024, escolhe em 2025, avalia
em 2026 sem tocar no teste), com uma diferença: a validação também é degradada,
porque o limiar precisa valer para relatos incompletos, que é onde o modelo
será usado.

Compara três limiares no teste 2026, em vários níveis de omissão por ligação:
- completo:  o 0.391 do `threshold.py`, escolhido em validação sem omissão;
- mistura:   escolhido numa validação com omissão sorteada em U(0, 1), como no
             treino. É o candidato a limiar único de produção;
- por nível: escolhido numa validação com o mesmo p do teste. Só seria possível
             se soubéssemos a taxa de omissão; serve de teto para medir quanto
             o limiar único perde.

Uso:
    python src/threshold_nan.py
"""

import json

import numpy as np
import pandas as pd

from augmentation import LEVELS
from data_preprocessing import TRIAGE, load_dataset
from degradation import augment_nan, drop_per_call_nan
from threshold import FIT_YEARS, TARGET_RECALL, VALID_YEARS, metrics_at, threshold_for_recall
from train import MODELS_DIR, TEST_YEARS, make_pipeline, make_xgboost

N_REPEATS = 10  # sorteios de omissão por nível no teste


def fit_aug_nan(train_years: list[int], seed: int):
    X, y = load_dataset(train_years)
    X_aug = augment_nan(X, np.random.default_rng(seed))
    return make_pipeline(TRIAGE, make_xgboost(y)).fit(X_aug, y)


def proba_at(model, X: pd.DataFrame, p: float, seed: int) -> np.ndarray:
    return model.predict_proba(drop_per_call_nan(X, p, np.random.default_rng(seed)))[:, 1]


def main() -> None:
    X_valid, y_valid = load_dataset(VALID_YEARS)
    model = fit_aug_nan(FIT_YEARS, seed=0)

    # Limiar por nível e o da mistura (cada acidente com seu p sorteado).
    by_level = {p: threshold_for_recall(y_valid, proba_at(model, X_valid, p, 1), TARGET_RECALL)
                for p in LEVELS}
    mixture = threshold_for_recall(
        y_valid, model.predict_proba(augment_nan(X_valid, np.random.default_rng(1)))[:, 1],
        TARGET_RECALL)
    complete = json.loads((MODELS_DIR / "xgboost_threshold.json").read_text())["threshold"]

    print(f"Limiares escolhidos na validação {VALID_YEARS} (recall >= {TARGET_RECALL:.0%}):")
    print(f"  completo (threshold.py): {complete:.3f}")
    print(f"  mistura U(0,1):          {mixture:.3f}")
    print("  por nível: " + ", ".join(f"p={p:.0%}: {t:.3f}" for p, t in by_level.items()), "\n")

    # Retreina em 2024+2025 (como no threshold.py) e avalia em 2026.
    X_test, y_test = load_dataset(TEST_YEARS)
    final = fit_aug_nan(FIT_YEARS + VALID_YEARS, seed=0)
    rows = []
    for p in LEVELS:
        for rep in range(N_REPEATS):
            proba = proba_at(final, X_test, p, 1000 + rep)
            for name, thr in [("completo", complete), ("mistura", mixture),
                              ("por nível", by_level[p])]:
                rows.append({"p": p, "limiar": name, **metrics_at(y_test, proba, thr)})
    results = pd.DataFrame(rows).groupby(["p", "limiar"]).mean().drop(columns="Limiar")

    for metric in ["Recall", "Precision", "Taxa de acionamento"]:
        print(metric)
        print(results[metric].unstack("limiar")[["completo", "mistura", "por nível"]]
              .round(3).to_string(), "\n")

    (MODELS_DIR / "xgboost_nan_threshold.json").write_text(json.dumps({
        "threshold": mixture, "target_recall": TARGET_RECALL, "chosen_on": VALID_YEARS,
        "validation": "omissão por ligação sorteada em U(0,1), bloco omitido = NaN"}))


if __name__ == "__main__":
    main()
