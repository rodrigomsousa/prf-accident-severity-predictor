"""Agregação dos arquivos por pessoa: contagem por veículo distinto, teto 5, pedestre."""

import pandas as pd

import data_preprocessing as dp


def test_load_people_features(tmp_path, monkeypatch):
    rows = [
        # id 1: duas motos (uma com passageiro) + um caminhão
        (1, 10, "Motocicleta", "Condutor"), (1, 10, "Motocicleta", "Passageiro"),
        (1, 11, "Motoneta", "Condutor"), (1, 12, "Caminhão", "Condutor"),
        # id 2: só um pedestre (sem veículo)
        (2, None, None, "Pedestre"),
        # id 3: sete bicicletas (teto 5)
        *[(3, 20 + i, "Bicicleta", "Condutor") for i in range(7)],
    ]
    pd.DataFrame(rows, columns=["id", "id_veiculo", "tipo_veiculo", "tipo_envolvido"]).to_csv(
        tmp_path / "pessoas2099.csv", sep=";", index=False, encoding="latin1")
    monkeypatch.setattr(dp, "RAW_DIR", tmp_path)

    out = dp.load_people_features([2099])
    assert out.loc[1, "n_moto"] == 2          # passageiro não é outra moto
    assert out.loc[1, "n_pesado"] == 1
    assert out.loc[1, "tem_pedestre"] == 0
    assert out.loc[2, "tem_pedestre"] == 1 and out.loc[2, "n_moto"] == 0
    assert out.loc[3, "n_bicicleta"] == 5     # teto
    assert list(out.columns) == dp.PEOPLE_COLS
