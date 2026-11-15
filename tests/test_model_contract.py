"""Contrato do pipeline: aceita NaN no relato, cross-fitting do TargetEncoder, ordem preservada."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder

from calibration import Isotonic, Platt, ece
from data_preprocessing import PEOPLE_COLS, TRIAGE
from threshold import threshold_for_recall
from train import make_pipeline, make_xgboost


def synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(index=range(n))
    for c in TRIAGE.categorical:
        X[c] = rng.choice(list("abcd"), n)
    for c in TRIAGE.numeric:
        X[c] = rng.integers(0, 4, n).astype(float)
    for c in TRIAGE.binary:
        X[c] = rng.integers(0, 2, n)
    for c in TRIAGE.target_encoded:
        X[c] = rng.choice([f"{c}{i}" for i in range(30)], n)
    y = pd.Series((X["n_moto"] + rng.normal(0, 1, n) > 2).astype(int))
    return X[TRIAGE.columns], y


@pytest.fixture(scope="module")
def fitted():
    X, y = synthetic()
    return make_pipeline(TRIAGE, make_xgboost(y)).fit(X, y), X, y


def test_inference_accepts_nan_in_report_block(fitted):
    model, X, _ = fitted
    Xn = X.copy()
    Xn[PEOPLE_COLS] = np.nan
    p = model.predict_proba(Xn)[:, 1]
    assert p.shape == (len(X),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_training_with_nan_block_works():
    X, y = synthetic()
    X = X.copy()
    X.loc[::2, PEOPLE_COLS] = np.nan
    model = make_pipeline(TRIAGE, make_xgboost(y)).fit(X, y)
    assert np.isfinite(model.predict_proba(X)[:, 1]).all()


def test_target_encoder_is_cross_fitted_inside_pipeline(fitted):
    model, X, y = fitted
    te = model.named_steps["prep"].named_transformers_["te"]
    assert isinstance(te, TargetEncoder) and isinstance(te.cv, KFold)
    # No treino, cada linha é codificada sem o próprio alvo (fit_transform != transform).
    X_te = X[TRIAGE.target_encoded]
    fresh = TargetEncoder(cv=KFold(5, shuffle=True, random_state=42))
    assert not np.allclose(fresh.fit_transform(X_te, y), fresh.transform(X_te))


def test_one_hot_handles_unseen_category(fitted):
    model, X, _ = fitted
    Xn = X.head(5).copy()
    Xn["uf"] = "zz"
    Xn["municipio"] = "nunca_visto"
    assert np.isfinite(model.predict_proba(Xn)[:, 1]).all()


def test_platt_preserves_ranking_and_fixes_calibration():
    rng = np.random.default_rng(0)
    p_true = rng.beta(2, 5, 20000)
    y = pd.Series((rng.random(20000) < p_true).astype(int))
    # Modelo que superestima por um deslocamento em log-odds (o erro que o Platt corrige,
    # como o do scale_pos_weight).
    p_inflated = 1 / (1 + np.exp(-(np.log(p_true / (1 - p_true)) + 1.0)))
    cal = Platt().fit(p_inflated, y).predict(p_inflated)
    order = np.argsort(p_inflated, kind="stable")
    assert (np.diff(cal[order]) >= -1e-12).all()  # monotônica: não muda ranking
    assert ece(y, cal) < ece(y, p_inflated) and ece(y, cal) < 0.02


def test_isotonic_output_in_unit_interval():
    rng = np.random.default_rng(1)
    p = rng.random(2000)
    y = pd.Series((rng.random(2000) < p).astype(int))
    out = Isotonic().fit(p, y).predict(p)
    assert ((out >= 0) & (out <= 1)).all()


def test_threshold_for_recall_meets_target_and_is_maximal():
    rng = np.random.default_rng(2)
    y = pd.Series(rng.integers(0, 2, 5000))
    p = np.clip(y * 0.3 + rng.random(5000) * 0.7, 0, 1)
    t = threshold_for_recall(y, p, 0.8)
    recall = (p[y == 1] >= t).mean()
    assert recall >= 0.8
    assert (p[y == 1] >= t + 0.01).mean() < 0.8  # um limiar maior já não cumpre a meta
