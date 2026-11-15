"""Omissão simulada do relato: o bloco de veículos/pessoas vira NaN ou 0, nada mais muda."""

import numpy as np
import pandas as pd
import pytest

from data_preprocessing import PEOPLE_COLS
from degradation import augment_nan, drop_per_call, drop_per_call_nan, drop_per_vehicle


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    n = 4000
    df = pd.DataFrame({c: rng.integers(1, 4, n) for c in PEOPLE_COLS})
    df["tem_pedestre"] = rng.integers(0, 2, n)
    df["km"] = rng.random(n) * 100
    df["uf"] = rng.choice(["SP", "MG"], n)
    return df


def test_augment_nan_hides_whole_block_or_nothing(X):
    out = augment_nan(X, np.random.default_rng(1))
    hidden = out[PEOPLE_COLS].isna()
    assert (hidden.all(axis=1) | ~hidden.any(axis=1)).all()  # tudo ou nada por ligação
    # taxa por ligação ~ U(0,1) => em média metade das ligações omitidas
    assert abs(hidden.all(axis=1).mean() - 0.5) < 0.05


def test_augment_does_not_touch_other_columns_or_input(X):
    before = X.copy()
    out = augment_nan(X, np.random.default_rng(1))
    pd.testing.assert_frame_equal(X, before)  # não muta a entrada
    pd.testing.assert_series_equal(out["km"], X["km"])
    pd.testing.assert_series_equal(out["uf"], X["uf"])


def test_visible_rows_keep_original_values(X):
    out = augment_nan(X, np.random.default_rng(1))
    keep = out[PEOPLE_COLS].notna().all(axis=1)
    assert (out.loc[keep, PEOPLE_COLS].to_numpy() == X.loc[keep, PEOPLE_COLS].to_numpy()).all()


def test_drop_per_call_levels(X):
    assert drop_per_call(X, 0.0, np.random.default_rng(0))[PEOPLE_COLS].eq(X[PEOPLE_COLS]).all().all()
    assert (drop_per_call(X, 1.0, np.random.default_rng(0))[PEOPLE_COLS] == 0).all().all()
    assert drop_per_call_nan(X, 1.0, np.random.default_rng(0))[PEOPLE_COLS].isna().all().all()
    assert drop_per_call_nan(X, 0.0, np.random.default_rng(0))[PEOPLE_COLS].notna().all().all()


def test_drop_per_vehicle_never_adds_vehicles(X):
    out = drop_per_vehicle(X, 0.5, np.random.default_rng(0))
    assert (out[PEOPLE_COLS].to_numpy() <= X[PEOPLE_COLS].to_numpy()).all()
