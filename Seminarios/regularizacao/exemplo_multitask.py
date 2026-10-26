import numpy as np
import pandas as pd
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


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


class CustomDataset(Dataset):
    def __init__(self, atributos, glicose, diabetes):
        self.atributos = torch.as_tensor(atributos, dtype=torch.float32)
        self.glicose = torch.as_tensor(glicose, dtype=torch.float32).reshape(-1, 1)
        self.diabetes = torch.as_tensor(diabetes, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.atributos)

    def __getitem__(self, indice):
        return self.atributos[indice], self.glicose[indice], self.diabetes[indice]


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


def gerar_dataset(n_amostras=N_AMOSTRAS):
    gerador = np.random.default_rng(SEMENTE)
    dados = pd.DataFrame(
        {
            "idade": gerador.uniform(18, 80, n_amostras),
            "imc": np.clip(gerador.normal(27, 5, n_amostras), 16, 45),
            "pressao_sistolica": np.clip(
                gerador.normal(122, 17, n_amostras), 85, 200
            ),
            "horas_exercicio": gerador.uniform(0, 10, n_amostras),
            "historico_familiar": gerador.binomial(1, 0.3, n_amostras),
        }
    )

    atributos_continuos = [
        "idade",
        "imc",
        "pressao_sistolica",
        "horas_exercicio",
    ]
    atributos_padronizados = (
        dados[atributos_continuos] - dados[atributos_continuos].mean()
    ) / dados[atributos_continuos].std(ddof=0)
    fator_metabolico = (
        0.3 * atributos_padronizados["idade"]
        + 0.4 * atributos_padronizados["imc"]
        + 0.3 * atributos_padronizados["pressao_sistolica"]
        - 0.4 * atributos_padronizados["horas_exercicio"]
        + 0.5 * dados["historico_familiar"]
    )
    glicose = 90 + 15 * fator_metabolico + gerador.normal(0, 8, n_amostras)
    probabilidade_diabetes = 1 / (1 + np.exp(-fator_metabolico))
    diabetes = (probabilidade_diabetes > 0.5).astype(np.int64)

    return dados, glicose.to_numpy(dtype=np.float32), diabetes.to_numpy()


def preparar_dados():
    dados, glicose, diabetes = gerar_dataset()
    indices = np.arange(len(dados))
    indices_treino, indices_temporarios = train_test_split(
        indices,
        test_size=0.3,
        random_state=SEMENTE,
        stratify=diabetes,
    )
    indices_validacao, indices_teste = train_test_split(
        indices_temporarios,
        test_size=0.5,
        random_state=SEMENTE,
        stratify=diabetes[indices_temporarios],
    )

    dados_atributos = dados.to_numpy(dtype=np.float32)
    colunas_continuas = [0, 1, 2, 3]
    scaler_atributos = StandardScaler()
    scaler_atributos.fit(dados_atributos[indices_treino][:, colunas_continuas])

    atributos_escalados = dados_atributos.copy()
    atributos_escalados[:, colunas_continuas] = scaler_atributos.transform(
        dados_atributos[:, colunas_continuas]
    )

    scaler_glicose = StandardScaler()
    scaler_glicose.fit(glicose[indices_treino].reshape(-1, 1))
    glicose_escalada = scaler_glicose.transform(glicose.reshape(-1, 1))

    conjuntos = []
    for indices_conjunto in (indices_treino, indices_validacao, indices_teste):
        conjuntos.append(
            CustomDataset(
                atributos_escalados[indices_conjunto],
                glicose_escalada[indices_conjunto],
                diabetes[indices_conjunto],
            )
        )

    gerador_lotes = torch.Generator().manual_seed(SEMENTE)
    carregadores = [
        DataLoader(
            conjuntos[0],
            batch_size=TAMANHO_LOTE,
            shuffle=True,
            generator=gerador_lotes,
        ),
        DataLoader(conjuntos[1], batch_size=TAMANHO_LOTE, shuffle=False),
        DataLoader(conjuntos[2], batch_size=TAMANHO_LOTE, shuffle=False),
    ]
    return carregadores, scaler_glicose


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

    carregadores, scaler_glicose = preparar_dados()
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