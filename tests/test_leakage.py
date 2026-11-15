"""Garantias contra data leakage: as decisões que o CLAUDE.md marca como fixas."""

import numpy as np
import pandas as pd

from data_preprocessing import (LEAKAGE_COLS, PEOPLE_COLS, PREVENTIVE, TRIAGE, TRIAGE_BASE,
                                build_target)

FORBIDDEN = set(LEAKAGE_COLS) | {"causa_acidente", "estado_fisico", "ilesos", "feridos"}


def test_no_outcome_column_in_any_feature_set():
    for fs in (PREVENTIVE, TRIAGE_BASE, TRIAGE):
        assert not FORBIDDEN & set(fs.columns)


def test_feature_sets_are_nested():
    assert set(PREVENTIVE.columns) < set(TRIAGE_BASE.columns) < set(TRIAGE.columns)
    assert set(TRIAGE.columns) - set(TRIAGE_BASE.columns) == set(PEOPLE_COLS)


def test_preventive_has_no_report_information():
    # Só contexto pré-evento: nada que o solicitante relata na ligação.
    report = {"tipo_acidente", "veiculos", "pessoas", *PEOPLE_COLS}
    assert not report & set(PREVENTIVE.columns)


def test_high_cardinality_columns_are_target_encoded_not_onehot():
    assert set(TRIAGE.target_encoded) == {"municipio", "trecho"}
    assert not {"municipio", "trecho"} & set(TRIAGE.categorical)


def test_target_is_death_or_serious_injury_only():
    df = pd.DataFrame({"mortos": [0, 1, 0, 0], "feridos_graves": [0, 0, 2, 0],
                       "feridos_leves": [5, 0, 0, 0]})
    assert build_target(df).tolist() == [0, 1, 1, 0]  # ferido leve não é grave


def test_all_features_are_unique():
    assert len(TRIAGE.columns) == len(set(TRIAGE.columns))
    assert not np.intersect1d(TRIAGE.categorical, TRIAGE.numeric).size
