"""Treina e avalia o modelo de triagem.

Divisão temporal: treina com os anos anteriores e testa no ano mais recente,
simulando o uso real (o modelo prevê acidentes que ainda não viu).

Uso:
    python src/train.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_preprocessing import TRIAGE, FeatureSet, load_dataset

TRAIN_YEARS = [2024, 2025]
TEST_YEARS = [2026]
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def make_pipeline(features: FeatureSet, estimator: ClassifierMixin) -> Pipeline:
    preprocessor = ColumnTransformer([
        # min_frequency agrupa categorias raras (< 50 ocorrências) em "infrequent";
        # handle_unknown evita erro com categorias novas no teste.
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50),
         features.categorical),
        ("num", StandardScaler(), features.numeric),
        ("bin", "passthrough", features.binary),
    ])
    return Pipeline([("prep", preprocessor), ("model", estimator)])


def make_logistic_regression() -> LogisticRegression:
    # class_weight="balanced" dá mais peso à classe minoritária (graves),
    # senão o modelo tende a quase nunca prever "grave".
    return LogisticRegression(class_weight="balanced", max_iter=2000)


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "Acurácia": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "ROC-AUC": roc_auc_score(y, y_proba),
        "PR-AUC": average_precision_score(y, y_proba),
    }


def main() -> None:
    X_train, y_train = load_dataset(TRAIN_YEARS)
    X_test, y_test = load_dataset(TEST_YEARS)
    print(f"Treino {TRAIN_YEARS}: {len(X_train):,} acidentes ({y_train.mean():.1%} graves)")
    print(f"Teste  {TEST_YEARS}: {len(X_test):,} acidentes ({y_test.mean():.1%} graves)\n")

    models = {
        # Referência mínima: sempre prevê a classe mais frequente.
        "Dummy (classe majoritária)": make_pipeline(
            TRIAGE, DummyClassifier(strategy="most_frequent")),
        "Logistic Regression": make_pipeline(TRIAGE, make_logistic_regression()),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_test, y_test)
        print(f"== {name}")
        print(confusion_matrix(y_test, model.predict(X_test)), "\n")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(models["Logistic Regression"], MODELS_DIR / "logistic_regression.joblib")

    print(pd.DataFrame(results).T.round(3).to_string())


if __name__ == "__main__":
    main()
