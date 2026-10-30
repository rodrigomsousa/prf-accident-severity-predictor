"""Simulação de relato incompleto nas features de veículos e pessoas.

As colunas de `PEOPLE_COLS` vêm do registro final da PRF; na ligação do 191 o
solicitante pode esquecer veículos ou não saber de nenhum. Estas funções
simulam essa omissão, tanto para testar a robustez (taxa `p` fixa) quanto para
treinar com omissão (`augment_*`, taxa `p` sorteada por acidente).

O valor omitido vira 0 (o modelo lê "não há moto") ou NaN ("não informado").
"""

import numpy as np
import pandas as pd

from data_preprocessing import PEOPLE_COLS


def drop_per_vehicle(X: pd.DataFrame, p: float, rng: np.random.Generator) -> pd.DataFrame:
    """Cada veículo (e o pedestre) é omitido com probabilidade p; o omitido vira 0."""
    X = X.copy()
    for col in PEOPLE_COLS:
        # Binomial(n, 1-p): de n veículos, cada um é lembrado com chance 1-p.
        # Para `tem_pedestre` (0/1) isso vale 1 com chance 1-p, como esperado.
        X[col] = rng.binomial(X[col].astype(int), 1 - p)
    return X


def drop_per_call(X: pd.DataFrame, p: float, rng: np.random.Generator) -> pd.DataFrame:
    """Com probabilidade p, a ligação inteira perde o bloco de veículos/pessoas (vira 0)."""
    X = X.copy()
    hidden = rng.random(len(X)) < p
    X.loc[hidden, PEOPLE_COLS] = 0
    return X


def hide_block_nan(X: pd.DataFrame, hidden: np.ndarray) -> pd.DataFrame:
    X = X.copy()
    X[PEOPLE_COLS] = X[PEOPLE_COLS].astype(float)
    X.loc[hidden, PEOPLE_COLS] = np.nan
    return X


def drop_per_call_nan(X: pd.DataFrame, p: float, rng: np.random.Generator) -> pd.DataFrame:
    """Como `drop_per_call`, mas o bloco omitido vira NaN ("não informado")."""
    return hide_block_nan(X, rng.random(len(X)) < p)


# Augmentation: cada acidente aparece uma única vez, com taxa de omissão própria
# sorteada em U(0, 1). Empilhar uma cópia degradada ao treino original repetiria
# a mesma linha em folds diferentes do TargetEncoder e vazaria o alvo.

def augment_zero(X: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Omissão por veículo com taxa p própria de cada acidente."""
    X = X.copy()
    p = rng.random(len(X))
    for col in PEOPLE_COLS:
        X[col] = rng.binomial(X[col].astype(int), 1 - p)
    return X


def augment_nan(X: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Omissão por ligação com taxa p própria; o bloco omitido vira NaN."""
    return hide_block_nan(X, rng.random(len(X)) < rng.random(len(X)))
