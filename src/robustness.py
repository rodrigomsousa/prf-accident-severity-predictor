"""Teste de robustez ao relato incompleto (veículos e pessoas).

As features `n_moto`, `n_bicicleta`, `n_pesado`, `n_onibus` e `tem_pedestre` vêm
do registro final da PRF, não do que o solicitante diz na ligação. Aqui
degradamos essas colunas *só no teste* e medimos quanto do ganho sobrevive.
O modelo é treinado uma única vez, com a informação completa.

Dois cenários de degradação, ambos simulando "o solicitante não mencionou":
- por veículo: cada veículo/pedestre é esquecido com probabilidade p,
  independentemente (o relato é parcial, mas com alguma informação);
- por ligação: com probabilidade p a ligação inteira chega sem esse bloco
  (o atendente não perguntou ou o solicitante não soube de nada).

O valor "esquecido" vira 0, porque é assim que o modelo lê a ausência: ele não
distingue "não há moto" de "ninguém falou da moto". É o cenário mais exigente.

Uso:
    python src/robustness.py
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from data_preprocessing import TRIAGE, TRIAGE_BASE, load_dataset
from degradation import drop_per_call, drop_per_vehicle
from train import MODELS_DIR, TEST_YEARS, TRAIN_YEARS, make_pipeline, make_xgboost

LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_REPEATS = 20  # repetições do sorteio por nível, para medir a variação


SCENARIOS = {"por veículo": drop_per_vehicle, "por ligação": drop_per_call}


def score(y: pd.Series, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y, proba),
        "PR-AUC": average_precision_score(y, proba),
        "Recall": recall_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Acionamento": pred.mean(),
    }


def run() -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    X_train, y_train = load_dataset(TRAIN_YEARS)
    X_test, y_test = load_dataset(TEST_YEARS)
    threshold = json.loads((MODELS_DIR / "xgboost_threshold.json").read_text())["threshold"]

    model = make_pipeline(TRIAGE, make_xgboost(y_train)).fit(X_train, y_train)
    # Piso: o mesmo modelo treinado sem nenhuma feature de veículos/pessoas.
    base = make_pipeline(TRIAGE_BASE, make_xgboost(y_train)).fit(X_train, y_train)
    reference = {
        "completo": score(y_test, model.predict_proba(X_test)[:, 1], threshold),
        "sem veículos/pessoas": score(y_test, base.predict_proba(X_test)[:, 1], threshold),
    }

    rows = []
    for s, (scenario, degrade) in enumerate(SCENARIOS.items()):
        for p in LEVELS:
            for rep in range(N_REPEATS):
                rng = np.random.default_rng([s, round(p * 100), rep])
                proba = model.predict_proba(degrade(X_test, p, rng))[:, 1]
                rows.append({"cenário": scenario, "p": p, "rep": rep,
                             **score(y_test, proba, threshold)})
    return pd.DataFrame(rows), reference


def main() -> None:
    results, reference = run()
    print("Referências no teste (limiar fixo do threshold.py):")
    print(pd.DataFrame(reference).T.round(3).to_string(), "\n")

    metrics = ["ROC-AUC", "PR-AUC", "Recall", "Precision", "Acionamento"]
    summary = results.groupby(["cenário", "p"])[metrics].agg(["mean", "std"]).round(3)
    for scenario in SCENARIOS:
        print(f"Cenário: omissão {scenario} (média e desvio em {N_REPEATS} sorteios)")
        print(summary.loc[scenario].to_string(), "\n")

    out = MODELS_DIR.parent / "data" / "robustness_results.csv"
    results.to_csv(out, index=False)
    print(f"Resultados brutos salvos em {out}")


if __name__ == "__main__":
    main()
