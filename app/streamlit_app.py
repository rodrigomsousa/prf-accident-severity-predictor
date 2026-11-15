"""Demo de triagem do 191: o atendente descreve a ocorrência e vê a recomendação.

Mostra a probabilidade calibrada de acidente grave, a decisão (acionar ou não
resgate avançado) e o que mais pesou na previsão (SHAP). Usa o modelo de
produção (`xgboost_aug_nan`), o calibrador Platt OOF e o limiar de recall 80%.

Precisa dos CSVs em `data/raw/` (para listar municípios, categorias e
coordenadas típicas de cada trecho) e dos modelos treinados em `models/`.

Uso (a partir da raiz do projeto):
    streamlit run app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from data_preprocessing import PEOPLE_COLS, TRACADO_TOKENS, TRIAGE, build_features, load_raw
from explain import shap_by_variable

MODELS = ROOT / "models"
PEOPLE_LABELS = {"n_moto": "Motos", "n_bicicleta": "Bicicletas", "n_pesado": "Pesados",
                 "n_onibus": "Ônibus"}
FEATURE_LABELS = {
    "tipo_acidente": "tipo de acidente", "pessoas": "nº de pessoas", "veiculos": "nº de veículos",
    "n_moto": "motos", "n_bicicleta": "bicicletas", "n_pesado": "caminhões/tratores",
    "n_onibus": "ônibus", "tem_pedestre": "pedestre", "municipio": "município",
    "trecho": "trecho da rodovia", "uf": "UF", "br": "rodovia", "km": "km", "hora": "hora",
    "dia_semana": "dia da semana", "mes": "mês", "fase_dia": "fase do dia",
    "sentido_via": "sentido da via", "condicao_metereologica": "clima", "tipo_pista": "tipo de pista",
    "uso_solo": "área (urbana/rural)", "latitude": "latitude", "longitude": "longitude",
}


@st.cache_resource(show_spinner="Carregando modelo e dados...")
def load_assets():
    model = joblib.load(MODELS / "xgboost_aug_nan.joblib")
    platt = joblib.load(MODELS / "xgboost_nan_platt_prod.joblib")
    threshold = json.loads((MODELS / "xgboost_nan_threshold.json").read_text())["threshold"]
    feats = build_features(load_raw([2024, 2025]))
    return model, platt, threshold, feats


def options(feats: pd.DataFrame, col: str) -> list[str]:
    return sorted(feats[col].dropna().unique())


def pick(container, label: str, feats: pd.DataFrame, col: str, **kw):
    """Selectbox com as categorias em ordem alfabética e a mais frequente como padrão."""
    opts = options(feats, col)
    return container.selectbox(label, opts, index=opts.index(feats[col].mode()[0]), **kw)


def typical_coords(feats: pd.DataFrame, trecho: str, municipio: str) -> tuple[float, float]:
    """Coordenadas medianas do trecho; se nunca visto, do município; senão NaN."""
    for col, key in [("trecho", trecho), ("municipio", municipio)]:
        sub = feats[feats[col] == key]
        if len(sub):
            return float(sub["latitude"].median()), float(sub["longitude"].median())
    return np.nan, np.nan


def main() -> None:
    st.set_page_config(page_title="Triagem 191 PRF", page_icon="🚑", layout="wide")
    st.title("Triagem de acidentes — 191 PRF")
    st.caption("Estima a chance de vítima grave ou fatal para apoiar o despacho de resgate "
               "avançado. Projeto de estudo; não é uma ferramenta operacional.")
    model, platt, threshold_raw, feats = load_assets()
    threshold = float(platt.predict(np.array([threshold_raw]))[0])

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Ocorrência")
        c1, c2, c3 = st.columns(3)
        uf = pick(c1, "UF", feats, "uf")
        brs = list(feats.loc[feats["uf"] == uf, "br"].value_counts().index)  # mais frequente primeiro
        br = c2.selectbox("Rodovia (BR)", brs)
        km = c3.number_input("Km", min_value=0.0, max_value=5000.0, value=100.0, step=1.0)
        in_uf = feats[(feats["uf"] == uf) & (feats["br"] == br)]
        municipios = list(in_uf["municipio"].value_counts().index) or options(feats, "municipio")
        municipio = st.selectbox("Município", municipios,
                                 format_func=lambda m: m.split("_", 1)[1].title())

        c1, c2, c3, c4 = st.columns(4)
        hora = c1.selectbox("Hora", [f"{h:02d}" for h in range(24)], index=18)
        dia = pick(c2, "Dia da semana", feats, "dia_semana")
        mes = c3.selectbox("Mês", [str(m) for m in range(1, 13)], index=10)
        fase = pick(c4, "Fase do dia", feats, "fase_dia")

        c1, c2 = st.columns(2)
        clima = pick(c1, "Condição meteorológica", feats, "condicao_metereologica")
        pista = pick(c2, "Tipo de pista", feats, "tipo_pista")
        c1, c2 = st.columns(2)
        sentido = pick(c1, "Sentido da via", feats, "sentido_via")
        uso = pick(c2, "Área", feats, "uso_solo")
        tracado = st.multiselect("Traçado da via", TRACADO_TOKENS, default=["Reta"])

        st.subheader("Relato do solicitante")
        tipo = pick(st, "Tipo de acidente", feats, "tipo_acidente")
        c1, c2 = st.columns(2)
        veiculos = c1.number_input("Veículos envolvidos", 0, 30, 2)
        pessoas = c2.number_input("Pessoas envolvidas", 0, 100, 3)
        informou = st.checkbox("O solicitante informou quais veículos estão envolvidos", value=True)
        people = dict.fromkeys(PEOPLE_COLS, np.nan)
        if informou:
            cols = st.columns(4)
            for col, (name, label) in zip(cols, PEOPLE_LABELS.items()):
                people[name] = col.number_input(label, 0, 5, 0, key=name)
            people["tem_pedestre"] = int(st.checkbox("Há pedestre envolvido"))

    lat, lon = typical_coords(feats, f"{br}_{uf}_{int(km // 10)}", municipio)
    row = {"uf": uf, "br": br, "dia_semana": dia, "hora": hora, "mes": mes, "fase_dia": fase,
           "sentido_via": sentido, "condicao_metereologica": clima, "tipo_pista": pista,
           "uso_solo": uso, "tipo_acidente": tipo, "km": km, "latitude": lat, "longitude": lon,
           "veiculos": veiculos, "pessoas": pessoas, "municipio": municipio,
           "trecho": f"{br}_{uf}_{int(km // 10)}", **people}
    row.update({f"tracado_{t}": int(t in tracado) for t in TRACADO_TOKENS})
    X = pd.DataFrame([row])[TRIAGE.columns]

    proba = float(platt.predict(model.predict_proba(X)[:, 1])[0])
    with right:
        st.subheader("Recomendação")
        acionar = proba >= threshold
        st.metric("Chance de vítima grave ou fatal", f"{proba:.0%}")
        if acionar:
            st.error(f"**ACIONAR resgate avançado** (limiar {threshold:.0%})")
        else:
            st.success(f"Resgate avançado não indicado (limiar {threshold:.0%})")
        st.caption(f"Base: {feats.shape[0]:,} acidentes de 2024–2025. Em média ~28% dos acidentes "
                   "são graves. O limiar foi escolhido para pegar ~80% dos graves "
                   "(custo: acionar em ~58% das ligações).")
        if not informou:
            st.warning("Sem o relato de veículos, o modelo usa só tipo de acidente e contexto: "
                       "tende a acionar mais, com menos precisão.")

        contrib, _ = shap_by_variable(model, X)
        c = contrib.iloc[0]
        top = c.reindex(c.abs().sort_values(ascending=False).index).head(8)[::-1]
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.barh([FEATURE_LABELS.get(n, n) for n in top.index], top.values,
                color=["#e34948" if v > 0 else "#2a78d6" for v in top.values])
        ax.axvline(0, color="#898781", linewidth=0.8)
        ax.set_xlabel("Contribuição (log-odds): vermelho aumenta o risco")
        ax.set_title("O que mais pesou nesta previsão")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


main()
