import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, atributos, glicose, diabetes):
        self.atributos = torch.as_tensor(atributos, dtype=torch.float32)
        self.glicose = torch.as_tensor(glicose, dtype=torch.float32).reshape(-1, 1)
        self.diabetes = torch.as_tensor(diabetes, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.atributos)

    def __getitem__(self, indice):
        return self.atributos[indice], self.glicose[indice], self.diabetes[indice]


def preparar_dados(n_amostras, semente, tamanho_lote):
    gerador = np.random.default_rng(semente)
    dados = pd.DataFrame({
        "idade": gerador.uniform(18, 80, n_amostras),
        "imc": np.clip(gerador.normal(27, 5, n_amostras), 16, 45),
        "pressao_sistolica": np.clip(gerador.normal(122, 17, n_amostras), 85, 200),
        "horas_exercicio": gerador.uniform(0, 10, n_amostras),
        "historico_familiar": gerador.binomial(1, 0.3, n_amostras),
    })
    continuos = ["idade", "imc", "pressao_sistolica", "horas_exercicio"]
    padronizados = (dados[continuos] - dados[continuos].mean()) / dados[continuos].std(ddof=0)
    fator = (0.3 * padronizados["idade"] + 0.4 * padronizados["imc"]
             + 0.3 * padronizados["pressao_sistolica"]
             - 0.4 * padronizados["horas_exercicio"]
             + 0.5 * dados["historico_familiar"])
    glicose = (90 + 15 * fator + gerador.normal(0, 8, n_amostras)).to_numpy(np.float32)
    diabetes = (1 / (1 + np.exp(-fator)) > 0.5).astype(np.int64).to_numpy()

    indices = np.arange(n_amostras)
    indices_treino, indices_temporarios = train_test_split(
        indices, test_size=0.3, random_state=semente, stratify=diabetes
    )
    indices_validacao, indices_teste = train_test_split(
        indices_temporarios, test_size=0.5, random_state=semente,
        stratify=diabetes[indices_temporarios]
    )
    atributos = dados.to_numpy(dtype=np.float32)
    scaler_atributos = StandardScaler().fit(atributos[indices_treino, :4])
    atributos[:, :4] = scaler_atributos.transform(atributos[:, :4])
    scaler_glicose = StandardScaler().fit(glicose[indices_treino].reshape(-1, 1))
    glicose = scaler_glicose.transform(glicose.reshape(-1, 1))
    conjuntos = [CustomDataset(atributos[indices], glicose[indices], diabetes[indices])
                 for indices in (indices_treino, indices_validacao, indices_teste)]
    gerador_lotes = torch.Generator().manual_seed(semente)
    carregadores = [
        DataLoader(conjuntos[0], batch_size=tamanho_lote, shuffle=True, generator=gerador_lotes),
        DataLoader(conjuntos[1], batch_size=tamanho_lote, shuffle=False),
        DataLoader(conjuntos[2], batch_size=tamanho_lote, shuffle=False),
    ]
    return carregadores, scaler_glicose
