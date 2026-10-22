"""Carregamento, limpeza e engenharia de features dos dados da PRF (datatran)."""

from dataclasses import dataclass, field
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
    # Categóricas com milhares de valores: viram a taxa histórica de graves
    # de cada categoria (target encoding) em vez de one-hot.
    target_encoded: list[str] = field(default_factory=list)

    @property
    def columns(self) -> list[str]:
        return self.categorical + self.numeric + self.binary + self.target_encoded


# Contexto conhecido *antes* do acidente: local, tempo, clima e via.
PREVENTIVE = FeatureSet(
    categorical=["uf", "br", "dia_semana", "hora", "mes", "fase_dia", "sentido_via",
                 "condicao_metereologica", "tipo_pista", "uso_solo"],
    numeric=["km", "latitude", "longitude"],
    binary=[f"tracado_{t}" for t in TRACADO_TOKENS],
    target_encoded=["municipio", "trecho"],
)

# Triagem no 191: contexto + o que o solicitante informa na ligação
# (tipo do acidente, quantos veículos e pessoas envolvidas).
# `causa_acidente` fica de fora: é definida depois, na investigação da PRF.
TRIAGE = FeatureSet(
    categorical=PREVENTIVE.categorical + ["tipo_acidente"],
    numeric=PREVENTIVE.numeric + ["veiculos", "pessoas"],
    binary=PREVENTIVE.binary,
    target_encoded=PREVENTIVE.target_encoded,
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

    for col in ["km", "veiculos", "pessoas", "latitude", "longitude"]:
        out[col] = df[col]

    # Prefixo da UF: há municípios homônimos em estados diferentes.
    out["municipio"] = df["uf"] + "_" + df["municipio"]
    # Trecho de 10 km da rodovia, ex.: "116_SP_23" = BR-116 em SP, km 230 a 239.
    out["trecho"] = (df["br"].astype(str) + "_" + df["uf"] + "_"
                     + (df["km"] // 10).astype(int).astype(str))

    # "Reta;Declive" -> tracado_Reta=1, tracado_Declive=1
    tracado = df["tracado_via"].fillna("").str.split(";")
    for token in TRACADO_TOKENS:
        out[f"tracado_{token}"] = tracado.apply(lambda ts: int(token in ts))

    return out


def load_dataset(years: list[int]) -> tuple[pd.DataFrame, pd.Series]:
    df = load_raw(years)
    return build_features(df), build_target(df)
