from pathlib import Path
import struct
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _ler_idx1(caminho):
    with open(caminho, "rb") as arquivo:
        magic, _ = struct.unpack(">II", arquivo.read(8))
        if magic != 2049:
            raise ValueError(f"Magic number invalido: {magic}")
        return np.fromfile(arquivo, dtype=np.uint8)


def _ler_idx3(caminho):
    with open(caminho, "rb") as arquivo:
        magic, quantidade, linhas, colunas = struct.unpack(">IIII", arquivo.read(16))
        if magic != 2051:
            raise ValueError(f"Magic number invalido: {magic}")
        return np.fromfile(arquivo, dtype=np.uint8).reshape(quantidade, linhas, colunas)


def _dataset(imagens, rotulos):
    entradas = torch.from_numpy(imagens.reshape(len(imagens), -1).astype(np.float32) / 255.0)
    alvos = torch.from_numpy(rotulos.astype(np.int64))
    return TensorDataset(entradas, alvos)


def carregar_dados(pasta_dados, tamanho_lote=128):
    pasta = Path(pasta_dados)
    treino = _dataset(_ler_idx3(pasta / "train-images.idx3-ubyte"), _ler_idx1(pasta / "train-labels.idx1-ubyte"))
    teste = _dataset(_ler_idx3(pasta / "t10k-images.idx3-ubyte"), _ler_idx1(pasta / "t10k-labels.idx1-ubyte"))
    gerador = torch.Generator().manual_seed(42)
    return (DataLoader(treino, batch_size=tamanho_lote, shuffle=True, generator=gerador),
            DataLoader(teste, batch_size=512, shuffle=False), treino, teste)


def criar_dataloader(dados, batch_size, shuffle=False, seed=None):
    parametros = {"batch_size": batch_size, "shuffle": shuffle}
    if seed is not None:
        parametros["generator"] = torch.Generator().manual_seed(seed)
    return DataLoader(dados, **parametros)


def adicionar_ruido_gaussiano(dados, quantidade_copias, seed, sigma, incluir_originais=False):
    imagens, rotulos = dados.tensors
    gerador = torch.Generator().manual_seed(seed)
    ruido = torch.randn((quantidade_copias, *imagens.shape), generator=gerador, dtype=imagens.dtype) * sigma
    imagens_com_ruido = ruido.add_(imagens.unsqueeze(0)).clamp_(0.0, 1.0).reshape(-1, imagens.shape[1])
    rotulos_com_ruido = rotulos.repeat(quantidade_copias)
    if incluir_originais:
        imagens_com_ruido = torch.cat((imagens, imagens_com_ruido))
        rotulos_com_ruido = torch.cat((rotulos, rotulos_com_ruido))
    return TensorDataset(imagens_com_ruido, rotulos_com_ruido)
