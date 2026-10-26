import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dataloaders.MultitaskLoader.multitask_loader import preparar_dados

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn


# Parâmetros do dataset e do treinamento
SEMENTE = 42
N_AMOSTRAS = 2000
N_NEURONIOS = 15
TAMANHO_LOTE = 100
EPOCAS = 150
TAXA_APRENDIZADO = 0.001
PESO_DECAY = 0
LAMBDA_CLASSIFICACAO = 1.0
ATIVACAO_COMPARTILHADA = nn.ReLU


class RedeMultitarefa(nn.Module):
    def __init__(self, n_atributos=5, n_neuronios=N_NEURONIOS):
        super().__init__()
        camadas_compartilhadas = []
        tamanho_entrada = n_atributos
        for _ in range(3):
            camadas_compartilhadas.extend(
                [
                    nn.Linear(tamanho_entrada, n_neuronios),
                    ATIVACAO_COMPARTILHADA(),
                ]
            )
            tamanho_entrada = n_neuronios
        self.tronco_compartilhado = nn.Sequential(*camadas_compartilhadas)

        self.cabeca_glicose = nn.Sequential(
            nn.Linear(n_neuronios, n_neuronios),
            nn.ReLU(),
            nn.Linear(n_neuronios, n_neuronios),
            nn.ReLU(),
            nn.Linear(n_neuronios, 1),
        )
        self.cabeca_diabetes = nn.Sequential(
            nn.Linear(n_neuronios, n_neuronios),
            nn.Sigmoid(),
            nn.Linear(n_neuronios, n_neuronios),
            nn.Sigmoid(),
            nn.Linear(n_neuronios, 1),
            nn.Sigmoid(),
        )

    def forward(self, atributos):
        representacao = self.tronco_compartilhado(atributos)
        return self.cabeca_glicose(representacao), self.cabeca_diabetes(
            representacao
        )


def avaliar_perdas(modelo, carregador, criterio_glicose, criterio_diabetes, dispositivo):
    modelo.eval()
    soma_perda_glicose = 0.0
    soma_perda_diabetes = 0.0
    total_amostras = 0

    with torch.no_grad():
        for atributos, glicose, diabetes in carregador:
            atributos = atributos.to(dispositivo)
            glicose = glicose.to(dispositivo)
            diabetes = diabetes.to(dispositivo)
            pred_glicose, pred_diabetes = modelo(atributos)
            tamanho_lote = len(atributos)
            soma_perda_glicose += (
                criterio_glicose(pred_glicose, glicose).item() * tamanho_lote
            )
            soma_perda_diabetes += (
                criterio_diabetes(pred_diabetes, diabetes).item() * tamanho_lote
            )
            total_amostras += tamanho_lote

    perda_glicose = soma_perda_glicose / total_amostras
    perda_diabetes = soma_perda_diabetes / total_amostras
    perda_total = perda_glicose + LAMBDA_CLASSIFICACAO * perda_diabetes
    return perda_glicose, perda_diabetes, perda_total


def treinar_modelo(carregador_treino, carregador_validacao):
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = RedeMultitarefa().to(dispositivo)
    criterio_glicose = nn.MSELoss()
    criterio_diabetes = nn.BCELoss()
    otimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=TAXA_APRENDIZADO,
        weight_decay=PESO_DECAY,
    )
    melhor_perda_validacao = float("inf")
    melhores_pesos = None
    largura_barra = 30
    intervalo_atualizacao = max(1, EPOCAS // 100)

    for epoca in range(EPOCAS):
        modelo.train()
        for atributos, glicose, diabetes in carregador_treino:
            atributos = atributos.to(dispositivo)
            glicose = glicose.to(dispositivo)
            diabetes = diabetes.to(dispositivo)

            otimizador.zero_grad()
            pred_glicose, pred_diabetes = modelo(atributos)
            perda_glicose = criterio_glicose(pred_glicose, glicose)
            perda_diabetes = criterio_diabetes(pred_diabetes, diabetes)
            perda_total = perda_glicose + LAMBDA_CLASSIFICACAO * perda_diabetes
            perda_total.backward()
            otimizador.step()

        perdas_validacao = avaliar_perdas(
            modelo,
            carregador_validacao,
            criterio_glicose,
            criterio_diabetes,
            dispositivo,
        )
        if perdas_validacao[2] < melhor_perda_validacao:
            melhor_perda_validacao = perdas_validacao[2]
            melhores_pesos = {
                nome: peso.detach().clone()
                for nome, peso in modelo.state_dict().items()
            }

        if (epoca + 1) % intervalo_atualizacao == 0 or epoca + 1 == EPOCAS:
            progresso = (epoca + 1) / EPOCAS
            preenchido = int(largura_barra * progresso)
            barra = "=" * preenchido + " " * (largura_barra - preenchido)
            print(
                f"\rTreinando rede multitarefa: [{barra}] "
                f"{progresso:6.1%} | perda de validação: "
                f"{perdas_validacao[2]:.6f}",
                end="\n" if epoca + 1 == EPOCAS else "",
                flush=True,
            )

    if melhores_pesos is not None:
        modelo.load_state_dict(melhores_pesos)
    return modelo, dispositivo


def avaliar_teste(modelo, carregador_teste, scaler_glicose, dispositivo):
    modelo.eval()
    previsoes_glicose = []
    previsoes_diabetes = []
    valores_glicose = []
    valores_diabetes = []

    with torch.no_grad():
        for atributos, glicose, diabetes in carregador_teste:
            pred_glicose, pred_diabetes = modelo(atributos.to(dispositivo))
            previsoes_glicose.extend(pred_glicose.cpu().numpy().ravel())
            previsoes_diabetes.extend(pred_diabetes.cpu().numpy().ravel())
            valores_glicose.extend(glicose.numpy().ravel())
            valores_diabetes.extend(diabetes.numpy().ravel())

    glicose_real = scaler_glicose.inverse_transform(
        np.asarray(valores_glicose).reshape(-1, 1)
    ).ravel()
    glicose_prevista = scaler_glicose.inverse_transform(
        np.asarray(previsoes_glicose).reshape(-1, 1)
    ).ravel()
    diabetes_real = np.asarray(valores_diabetes, dtype=np.int64)
    probabilidade_diabetes = np.asarray(previsoes_diabetes)
    diabetes_previsto = (probabilidade_diabetes >= 0.5).astype(np.int64)

    rmse = np.sqrt(mean_squared_error(glicose_real, glicose_prevista))
    mae = mean_absolute_error(glicose_real, glicose_prevista)
    acuracia = accuracy_score(diabetes_real, diabetes_previsto)
    precisao = precision_score(diabetes_real, diabetes_previsto, zero_division=0)
    revocacao = recall_score(diabetes_real, diabetes_previsto, zero_division=0)
    f1 = f1_score(diabetes_real, diabetes_previsto, zero_division=0)
    auc = roc_auc_score(diabetes_real, probabilidade_diabetes)

    print("\nResultados no conjunto de teste")
    print(f"Glicose | RMSE: {rmse:.2f} mg/dL | MAE: {mae:.2f} mg/dL")
    print(
        "Diabetes | "
        f"acurácia: {acuracia:.3f} | precisão: {precisao:.3f} | "
        f"revocação: {revocacao:.3f} | F1: {f1:.3f} | ROC-AUC: {auc:.3f}"
    )


def executar():
    np.random.seed(SEMENTE)
    torch.manual_seed(SEMENTE)

    carregadores, scaler_glicose = preparar_dados(
        N_AMOSTRAS, SEMENTE, TAMANHO_LOTE
    )
    print(
        "Amostras: "
        f"treino={len(carregadores[0].dataset)}, "
        f"validação={len(carregadores[1].dataset)}, "
        f"teste={len(carregadores[2].dataset)}"
    )
    modelo, dispositivo = treinar_modelo(carregadores[0], carregadores[1])
    avaliar_teste(modelo, carregadores[2], scaler_glicose, dispositivo)


if __name__ == "__main__":
    executar()