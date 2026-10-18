"""Carregamento, limpeza e engenharia de features dos dados da PRF (datatran)."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Colunas que descrevem o *resultado* do acidente. Usá-las como feature seria
# vazamento de dados (data leakage): o modelo "veria a resposta".
LEAKAGE_COLS = [
    "classificacao_acidente",
    "mortos",
    "feridos_leves",
    "feridos_graves",
    "feridos",
    "ilesos",
    "ignorados",
]

TRACADO_TOKENS = [
    "Reta", "Curva", "Declive", "Aclive", "Interseção de Vias", "Rotatória",
    "Retorno Regulamentado", "Em Obras", "Viaduto", "Ponte", "Desvio Temporário",
    "Túnel",
]

TARGET = "grave"


@dataclass(frozen=True)
class FeatureSet:
    categorical: list[str]
    numeric: list[str]
    binary: list[str]

    @property
    def columns(self) -> list[str]:
        return self.categorical + self.numeric + self.binary


# Contexto conhecido *antes* do acidente: local, tempo, clima e via.
PREVENTIVE = FeatureSet(
    categorical=["uf", "br", "dia_semana", "hora", "mes", "fase_dia", "sentido_via",
                 "condicao_metereologica", "tipo_pista", "uso_solo"],
    numeric=["km"],
    binary=[f"tracado_{t}" for t in TRACADO_TOKENS],
)

# Triagem no 191: contexto + o que o solicitante informa na ligação
# (tipo do acidente, quantos veículos e pessoas envolvidas).
# `causa_acidente` fica de fora: é definida depois, na investigação da PRF.
TRIAGE = FeatureSet(
    categorical=PREVENTIVE.categorical + ["tipo_acidente"],
    numeric=PREVENTIVE.numeric + ["veiculos", "pessoas"],
    binary=PREVENTIVE.binary,
)


def load_raw(years: list[int]) -> pd.DataFrame:
    """Lê os CSVs `datatran{ano}.csv` e concatena em um único DataFrame."""
    frames = []
    for year in years:
        df = pd.read_csv(
            RAW_DIR / f"datatran{year}.csv",
            sep=";",
            encoding="latin1",
            decimal=",",
        )
        df["ano"] = year
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_target(df: pd.DataFrame) -> pd.Series:
    """Acidente grave = ao menos um morto ou ferido grave."""
    return ((df["mortos"] > 0) | (df["feridos_graves"] > 0)).astype(int).rename(TARGET)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engenharia de features linha a linha (não aprende nada dos dados,
    então pode ser aplicada antes do split sem risco de leakage).
    Retorna todas as features candidatas; cada FeatureSet escolhe as suas."""
    out = pd.DataFrame(index=df.index)

    data = pd.to_datetime(df["data_inversa"])
    out["mes"] = data.dt.month.astype(str)
    out["hora"] = df["horario"].str[:2]

    for col in ["uf", "br", "dia_semana", "fase_dia", "sentido_via",
                "condicao_metereologica", "tipo_pista", "uso_solo",
                "tipo_acidente", "causa_acidente"]:
        out[col] = df[col].astype(str)

    for col in ["km", "veiculos", "pessoas"]:
        out[col] = df[col]

    # "Reta;Declive" -> tracado_Reta=1, tracado_Declive=1
    tracado = df["tracado_via"].fillna("").str.split(";")
    for token in TRACADO_TOKENS:
        out[f"tracado_{token}"] = tracado.apply(lambda ts: int(token in ts))

    return out


def load_dataset(years: list[int]) -> tuple[pd.DataFrame, pd.Series]:
    df = load_raw(years)
    return build_features(df), build_target(df)
