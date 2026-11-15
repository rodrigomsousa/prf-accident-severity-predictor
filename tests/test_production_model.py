"""Smoke test do modelo de produção salvo, no teste 2026 (pulado sem dados/modelos)."""

import json

import joblib
import pytest
from sklearn.metrics import roc_auc_score

from data_preprocessing import RAW_DIR, load_dataset
from threshold_nan import proba_at
from train import MODELS_DIR

NEEDED = [MODELS_DIR / n for n in ("xgboost_aug_nan.joblib", "xgboost_nan_threshold.json",
                                   "xgboost_nan_platt_prod.joblib")]
pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in NEEDED) or not (RAW_DIR / "datatran2026.csv").exists(),
    reason="precisa de data/raw e models/ locais")


@pytest.fixture(scope="module")
def scored():
    model = joblib.load(MODELS_DIR / "xgboost_aug_nan.joblib")
    platt = joblib.load(MODELS_DIR / "xgboost_nan_platt_prod.joblib")
    thr = json.loads((MODELS_DIR / "xgboost_nan_threshold.json").read_text())["threshold"]
    X, y = load_dataset([2026])
    return model, platt, thr, X, y


def test_recall_target_holds_on_test_year_at_any_omission(scored):
    model, _, thr, X, y = scored
    for p in (0.0, 0.5, 1.0):
        pred = proba_at(model, X, p, 0) >= thr
        assert pred[y.to_numpy() == 1].mean() > 0.78  # meta 80% com folga de amostragem


def test_calibrated_probabilities_match_base_rate(scored):
    model, platt, _, X, y = scored
    cal = platt.predict(proba_at(model, X, 0.0, 0))
    assert abs(cal.mean() - y.mean()) < 0.02


def test_model_beats_baseline(scored):
    model, _, _, X, y = scored
    assert roc_auc_score(y, proba_at(model, X, 0.0, 0)) > 0.75
    assert roc_auc_score(y, proba_at(model, X, 1.0, 0)) > 0.70  # sem nenhum relato
