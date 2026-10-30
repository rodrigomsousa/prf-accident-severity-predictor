"""Treino com omissão simulada (data augmentation) contra o relato incompleto.

O `robustness.py` mostrou que um modelo treinado só com relatos completos lê o
0 de "ninguém mencionou" como "não há moto" e erra. Aqui o próprio treino já vem
com omissão, para o modelo aprender que um 0 pode ser omissão.

Variantes (todas avaliadas no teste 2026, degradado como no robustness.py):
- completo:  treino sem omissão (o modelo atual);
- aug-zero:  treino com omissão por veículo, o omitido vira 0;
- aug-nan:   treino com omissão por ligação, o bloco omitido vira NaN
             ("não informado"). O atendente sabe que não sabe; o XGBoost
             trata NaN nativamente. Só vale para o cenário por ligação.

Cada acidente do treino aparece uma única vez, com taxa de omissão própria
sorteada em U(0, 1). Empilhar uma cópia degradada ao treino original repetiria
a mesma linha em folds diferentes do TargetEncoder e vazaria o alvo.

Uso:
    python src/augmentation.py
"""

import json

import numpy as np
import pandas as pd

from data_preprocessing import TRIAGE, load_dataset
from degradation import (augment_nan, augment_zero, drop_per_call, drop_per_call_nan,
                         drop_per_vehicle)
from robustness import N_REPEATS, score
from train import MODELS_DIR, TEST_YEARS, TRAIN_YEARS, make_pipeline, make_xgboost

LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
TARGET_RECALL = 0.80


def rate_at_recall(y: pd.Series, proba: np.ndarray, target: float) -> float:
    """Fração de ligações acionadas para atingir `target` de recall (limiar
    escolhido no próprio teste: serve só para comparar rankings, não para decidir)."""
    cutoff = np.quantile(proba[y.to_numpy() == 1], 1 - target)
    return float((proba >= cutoff).mean())


def evaluate(models: dict, X_test: pd.DataFrame, y_test: pd.Series, threshold: float) -> pd.DataFrame:
    # (modelo, cenário, função de degradação): o modelo aug-nan só é testado
    # com NaN, que é como o atendente registraria o "não sei".
    plan = [
        ("completo", "por veículo", drop_per_vehicle),
        ("aug-zero", "por veículo", drop_per_vehicle),
        ("completo", "por ligação", drop_per_call),
        ("aug-zero", "por ligação", drop_per_call),
        ("aug-nan", "por ligação", drop_per_call_nan),
    ]
    rows = []
    for s, (name, scenario, degrade) in enumerate(plan):
        for p in LEVELS:
            for rep in range(N_REPEATS):
                rng = np.random.default_rng([100 + s, round(p * 100), rep])
                proba = models[name].predict_proba(degrade(X_test, p, rng))[:, 1]
                rows.append({"modelo": name, "cenário": scenario, "p": p, "rep": rep,
                             **score(y_test, proba, threshold),
                             "Acion@R80": rate_at_recall(y_test, proba, TARGET_RECALL)})
    return pd.DataFrame(rows)


def main() -> None:
    X_train, y_train = load_dataset(TRAIN_YEARS)
    X_test, y_test = load_dataset(TEST_YEARS)
    threshold = json.loads((MODELS_DIR / "xgboost_threshold.json").read_text())["threshold"]

    aug_rng = np.random.default_rng(0)
    train_sets = {
        "completo": X_train,
        "aug-zero": augment_zero(X_train, aug_rng),
        "aug-nan": augment_nan(X_train, aug_rng),
    }
    models = {name: make_pipeline(TRIAGE, make_xgboost(y_train)).fit(X, y_train)
              for name, X in train_sets.items()}

    results = evaluate(models, X_test, y_test, threshold)
    for scenario in ["por veículo", "por ligação"]:
        sub = results[results["cenário"] == scenario]
        print(f"=== Omissão {scenario} (média de {N_REPEATS} sorteios)")
        for metric, title in [("PR-AUC", "PR-AUC"),
                              ("Recall", f"Recall no limiar fixo {threshold:.3f}"),
                              ("Acion@R80", "Acionamento necessário para recall 80%")]:
            table = sub.pivot_table(index="p", columns="modelo", values=metric, aggfunc="mean")
            print(f"\n{title}")
            print(table.round(3).to_string())
        print()

    out = MODELS_DIR.parent / "data" / "augmentation_results.csv"
    results.to_csv(out, index=False)
    print(f"Resultados brutos salvos em {out}")


if __name__ == "__main__":
    main()
