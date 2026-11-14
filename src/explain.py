"""Interpretabilidade do XGBoost `aug-nan` com valores SHAP.

SHAP reparte a previsão de cada ligação entre as variáveis: parte do valor médio
do modelo (base) e cada variável soma ou subtrai uma contribuição até chegar à
previsão final. As contribuições estão em log-odds (a escala antes da sigmoide),
então são aditivas: base + soma das contribuições = log-odds da previsão.

Em vez da biblioteca `shap`, usa o TreeSHAP nativo do XGBoost
(`pred_contribs=True`), que é exato para árvores e não exige dependência extra.

Como o pipeline expande categóricas em one-hot (`uf_SP`, `uf_MG`, ...), as
contribuições são somadas de volta à variável original (`uf`): por serem
aditivas, a soma continua sendo a contribuição daquela variável.

Usa o modelo salvo por `train.py` (2024 + 2025) e explica o teste 2026.

Uso:
    python src/explain.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from data_preprocessing import PEOPLE_COLS, TRIAGE, load_dataset
from degradation import hide_block_nan
from train import MODELS_DIR, TEST_YEARS

import matplotlib.pyplot as plt

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
N_SAMPLE = 20_000


def original_variable(feature_name: str) -> str:
    """'cat__uf_SP' -> 'uf'; 'num__km' -> 'km'; 'te__municipio' -> 'municipio'."""
    prefix, name = feature_name.split("__", 1)
    if prefix == "cat":
        # Escolhe a coluna categórica de nome mais longo que prefixa o nome.
        matches = [c for c in TRIAGE.categorical if name.startswith(c + "_")]
        return max(matches, key=len)
    return name


def shap_by_variable(model, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Contribuições SHAP (log-odds) por variável original e o valor-base."""
    prep, booster = model.named_steps["prep"], model.named_steps["model"].get_booster()
    Z = prep.transform(X)
    contribs = booster.predict(xgb.DMatrix(Z), pred_contribs=True)  # última coluna = base
    names = [original_variable(n) for n in prep.get_feature_names_out()]
    table = pd.DataFrame(contribs[:, :-1], columns=names, index=X.index)
    return table.T.groupby(level=0).sum().T, float(contribs[0, -1])


def importance(contrib: pd.DataFrame) -> pd.Series:
    return contrib.abs().mean().sort_values(ascending=False)


def plot_importance(imp: dict[str, pd.Series], top: int = 15) -> None:
    order = imp["completo"].head(top).index[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    height = 0.4
    for i, (label, series) in enumerate(imp.items()):
        ax.barh(np.arange(len(order)) + i * height, series.reindex(order).fillna(0),
                height=height, label=label)
    ax.set_yticks(np.arange(len(order)) + height / 2, order)
    ax.set_xlabel("Média de |SHAP| (log-odds)")
    ax.set_title("Importância das variáveis no XGBoost aug-nan (teste 2026)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "shap_importance.png", dpi=130)
    plt.close(fig)


def explain_call(model, row: pd.DataFrame, base: float, top: int = 6) -> None:
    contrib, _ = shap_by_variable(model, row)
    c = contrib.iloc[0]
    proba = model.predict_proba(row)[0, 1]
    print(f"  probabilidade bruta = {proba:.2f} (log-odds {base + c.sum():+.2f} = "
          f"base {base:+.2f} + soma das contribuições {c.sum():+.2f})")
    for name in c.abs().sort_values(ascending=False).head(top).index:
        value = row.iloc[0][name]
        print(f"    {name:18s} = {str(value)[:28]:28s} {c[name]:+.2f}")


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    model = joblib.load(MODELS_DIR / "xgboost_aug_nan.joblib")
    X, y = load_dataset(TEST_YEARS)
    X = X.sample(min(N_SAMPLE, len(X)), random_state=0)
    y = y.loc[X.index]

    full, base = shap_by_variable(model, X)
    hidden, _ = shap_by_variable(model, hide_block_nan(X, np.ones(len(X), dtype=bool)))
    imp = {"completo": importance(full), "sem relato de veículos/pessoas": importance(hidden)}
    plot_importance(imp)

    print(f"Valor-base (log-odds): {base:+.2f}  -> probabilidade {1 / (1 + np.exp(-base)):.2f}\n")
    table = pd.DataFrame(imp).sort_values("completo", ascending=False).round(3)
    print("Importância global (média de |SHAP|), top 15:")
    print(table.head(15).to_string(), "\n")
    print(f"Gráfico salvo em {REPORTS_DIR / 'shap_importance.png'}\n")

    # Efeito médio de cada grupo de veículos: o que o relato ganha ao informá-los.
    print("Contribuição média (log-odds) por valor, veículos/pessoas:")
    for col in PEOPLE_COLS:
        by_value = full[col].groupby(X[col].clip(upper=2)).mean().round(2)
        print(f"  {col:13s}", "  ".join(f"{k}: {v:+.2f}" for k, v in by_value.items()))

    # Casos individuais: o mais e o menos grave previstos entre os reais graves.
    proba = model.predict_proba(X)[:, 1]
    cases = {"maior probabilidade": proba.argmax(), "menor probabilidade": proba.argmin(),
             "grave que o modelo subestimou": np.where(y.to_numpy() == 1, proba, 2).argmin()}
    for label, pos in cases.items():
        print(f"\nLigação de 2026 - {label} (grave real = {y.iloc[pos]}):")
        explain_call(model, X.iloc[[pos]], base)


if __name__ == "__main__":
    main()
