import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SinDataset(Dataset):
    def __init__(self, entradas, alvos):
        self.entradas = torch.as_tensor(entradas, dtype=torch.float32).reshape(-1, 1)
        self.alvos = torch.as_tensor(alvos, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.entradas)

    def __getitem__(self, indice):
        return self.entradas[indice], self.alvos[indice]


def criar_dataloader(entradas, alvos, batch_size=None, shuffle=False):
    dataset = SinDataset(entradas, alvos)
    return DataLoader(dataset, batch_size=batch_size or len(dataset), shuffle=shuffle)


def criar_dataloaders(n_amostras, sigma, semente, fracao_treino=0.7):
    gerador = np.random.default_rng(semente)
    entradas = np.linspace(0, 2 * np.pi, n_amostras, dtype=np.float32)
    alvos = np.sin(entradas) + gerador.normal(0, sigma, n_amostras).astype(np.float32)
    indices = torch.randperm(n_amostras, generator=torch.Generator().manual_seed(semente + 1))
    tamanho_treino = int(fracao_treino * n_amostras)
    indices_treino = indices[:tamanho_treino]
    indices_validacao = indices[tamanho_treino:]
    dataset = SinDataset(entradas, alvos)
    return (
        DataLoader(torch.utils.data.Subset(dataset, indices_treino), batch_size=tamanho_treino, shuffle=False),
        DataLoader(torch.utils.data.Subset(dataset, indices_validacao), batch_size=len(indices_validacao), shuffle=False),
        DataLoader(dataset, batch_size=len(dataset), shuffle=False),
    )
