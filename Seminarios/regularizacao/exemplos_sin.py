from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


# Parâmetros do dataset e do treinamento
N = 20
sigma = 0.25
depth = 5
epochs = 1000
learning_rate = 0.01
lambda_l2 = 5e-3
lambda_l1 = 1.0082e-3
max_norm_coluna = 5e-1
epochs_entre_avaliacoes = 15
paciencia = 10

def criar_modelo(n, depth):
    camadas = []
    tamanho_entrada = 1
    ativacao = nn.Tanh()

    for _ in range(depth):
        camadas.extend([nn.Linear(tamanho_entrada, n), ativacao])
        tamanho_entrada = n

    camadas.append(nn.Linear(tamanho_entrada, 1))
    return nn.Sequential(*camadas)


def treinar_modelo(
    n, X, Y, depth, usar_l2=False, usar_l1=False, usar_projecao=False,
    X_validacao=None, Y_validacao=None, epochs_treinamento=None,
):
    # A mesma semente permite comparar os modelos de 15 neurônios
    # com a mesma inicialização.
    torch.manual_seed(123)

    modelo = criar_modelo(n, depth)
    if usar_l2:
        otimizador = torch.optim.Adam(
            modelo.parameters(),
            lr=learning_rate,
            weight_decay=lambda_l2,
        )
    else:
        otimizador = torch.optim.Adam(
            modelo.parameters(),
            lr=learning_rate,
        )
    criterio = nn.MSELoss()
    usar_parada_antecipada = X_validacao is not None
    numero_epochs = epochs if epochs_treinamento is None else epochs_treinamento

    regularizacao = (
        "L1" if usar_l1 else "L2" if usar_l2
        else "projeção L2 por coluna" if usar_projecao else "não"
    )
    nome_modelo = f"n={n}, regularização {regularizacao}"
    largura_barra = 30
    intervalo_atualizacao = max(1, numero_epochs // 100)

    melhor_estado = None
    melhor_erro_validacao = float("inf")
    melhor_epoch = 0
    melhor_avaliacao = 0
    numero_avaliacoes = 0
    avaliacoes_sem_melhora = 0
    if usar_parada_antecipada:
        with torch.no_grad():
            melhor_erro_validacao = criterio(
                modelo(X_validacao), Y_validacao
            ).item()
        melhor_estado = deepcopy(modelo.state_dict())

    for indice_epoch in range(numero_epochs):
        previsoes = modelo(X)
        perda = criterio(previsoes, Y)
        if usar_l1:
            penalidade_l1 = sum(
                camada.weight.abs().sum()
                for camada in modelo.modules()
                if isinstance(camada, nn.Linear)
            )
            perda = perda + lambda_l1 * penalidade_l1

        otimizador.zero_grad()
        perda.backward()
        otimizador.step()
        if usar_projecao:
            with torch.no_grad():
                for camada in modelo.modules():
                    if isinstance(camada, nn.Linear):
                        normas = torch.linalg.vector_norm(
                            camada.weight, ord=2, dim=0, keepdim=True
                        )
                        escalas = (max_norm_coluna / (normas + 1e-12)).clamp(max=1.0)
                        camada.weight.mul_(escalas)

        epoch_atual = indice_epoch + 1
        if usar_parada_antecipada and (
            epoch_atual % epochs_entre_avaliacoes == 0
            or epoch_atual == numero_epochs
        ):
            numero_avaliacoes += 1
            modelo.eval()
            with torch.no_grad():
                erro_validacao = criterio(
                    modelo(X_validacao), Y_validacao
                ).item()
            modelo.train()
           # print(f'Epoch {epoch_atual:4d} | Avaliação {numero_avaliacoes:3d} | Erro validação: {erro_validacao:.6f}')
            if erro_validacao < melhor_erro_validacao:
                melhor_erro_validacao = erro_validacao
                melhor_estado = deepcopy(modelo.state_dict())
                melhor_epoch = epoch_atual
                melhor_avaliacao = numero_avaliacoes
                avaliacoes_sem_melhora = 0
            else:
                avaliacoes_sem_melhora += 1
                if avaliacoes_sem_melhora >= paciencia:
                    print(
                        f"\nParada antecipada no epoch {epoch_atual}, "
                        f"na avaliação {numero_avaliacoes}; melhor validação "
                        f"na avaliação {melhor_avaliacao}, epoch {melhor_epoch} "
                        f"(MSE {melhor_erro_validacao:.6f})."
                    )
                    break

        if epoch_atual % intervalo_atualizacao == 0 or epoch_atual == numero_epochs:
            progresso = epoch_atual / numero_epochs
            preenchido = int(largura_barra * progresso)
            barra = "=" * preenchido + " " * (largura_barra - preenchido)
            print(
                f"\rTreinando {nome_modelo}: [{barra}] "
                f"{progresso:6.1%} | perda: {perda.item():.6f}",
                end="\n" if epoch_atual == numero_epochs else "",
                flush=True,
            )

    if usar_parada_antecipada:
        modelo.load_state_dict(melhor_estado)
        print(
            f"\nModelo restaurado da melhor validação: avaliação "
            f"{melhor_avaliacao}, epoch {melhor_epoch} "
            f"(MSE {melhor_erro_validacao:.6f})."
        )
        return modelo, melhor_epoch
    return modelo, numero_epochs


def plotar_modelos(
    indices, titulo_figura, figsize, modelos, x, y, x_plot, y_original, predicoes
):
    fig, eixos = plt.subplots(1, len(indices), figsize=figsize, sharey=True)
    eixos = np.atleast_1d(eixos)
    for eixo, indice in zip(eixos, indices):
        titulo, _, _ = modelos[indice]
        eixo.scatter(x, y, label="Dataset com ruído", color="tab:blue", alpha=0.7)
        eixo.plot(x_plot, y_original, label="Modelo gerador", color="tab:green")
        eixo.plot(x_plot, predicoes[indice], label="Modelo treinado", color="tab:red")
        eixo.set_title(titulo)
        eixo.set_xlabel("x")
        eixo.grid(True, alpha=0.3)
        eixo.legend()
    eixos[0].set_ylabel("y")
    fig.suptitle(titulo_figura)
    fig.tight_layout()
    return fig


def executar():
    # Reprodutibilidade
    np.random.seed(42)
    torch.manual_seed(42)

    # Dataset: pontos igualmente distribuídos entre 0 e 2*pi
    x = np.linspace(0, 2 * np.pi, N, dtype=np.float32)
    y = np.sin(x) + np.random.normal(0, sigma, N).astype(np.float32)

    X = torch.from_numpy(x).reshape(-1, 1)
    Y = torch.from_numpy(y).reshape(-1, 1)

    # A divisão 70/30 é exclusiva do modelo com parada antecipada.
    gerador_divisao = torch.Generator().manual_seed(43)
    indices = torch.randperm(X.size(0), generator=gerador_divisao)
    tamanho_treino = int(0.7 * X.size(0))
    indices_treino = indices[:tamanho_treino]
    indices_validacao = indices[tamanho_treino:]
    X_treino_parada = X[indices_treino]
    Y_treino_parada = Y[indices_treino]
    X_validacao = X[indices_validacao]
    Y_validacao = Y[indices_validacao]

    # Treina os seis modelos
    configuracoes = [
        (f"n=1, depth={depth}, epochs={epochs}", 1, False, False, False, False),
        (f"n=15, depth={depth}, epochs={epochs}", 15, False, False, False, False),
        (f"n=15, depth={depth}, epochs={epochs}, regularização L2", 15, True, False, False, False),
        (f"n=15, depth={depth}, epochs={epochs}, regularização L1", 15, False, True, False, False),
        (f"n=15, depth={depth}, projeção L2 por coluna", 15, False, False, True, False),
        (
            f"n=15, parada antecipada (epochs_entre_avaliacoes={epochs_entre_avaliacoes}, "
            f"paciência={paciencia})",
            15, False, False, False, True,
        ),
    ]

    modelos = []
    for titulo, n, usar_l2, usar_l1, usar_projecao, usar_parada in configuracoes:
        if usar_parada:
            _, melhor_epoch = treinar_modelo(
                n,
                X_treino_parada,
                Y_treino_parada,
                depth,
                usar_l2,
                usar_l1,
                usar_projecao,
                X_validacao,
                Y_validacao,
            )
            print(
                f"\nRetreinando do zero com todo o conjunto por "
                f"{melhor_epoch} epochs."
            )
            modelo, _ = treinar_modelo(
                n,
                X,
                Y,
                depth,
                usar_l2,
                usar_l1,
                usar_projecao,
                epochs_treinamento=melhor_epoch,
            )
            modelos.append((titulo, modelo, melhor_epoch))
        else:
            modelo, numero_epochs = treinar_modelo(
                n, X, Y, depth, usar_l2, usar_l1, usar_projecao
            )
            modelos.append((titulo, modelo, numero_epochs))

    # Predições em uma grade densa
    x_plot = np.linspace(0, 2 * np.pi, 500, dtype=np.float32)
    X_plot = torch.from_numpy(x_plot).reshape(-1, 1)
    y_original = np.sin(x_plot)
    Y_plot = torch.from_numpy(y_original).reshape(-1, 1)

    # Erros MSE dos modelos
    criterio_erro = nn.MSELoss()
    print("\nErros dos modelos (MSE)")
    print(f"{'Modelo':<52} {'Treinamento':>14} {'Validação':>14} {'Teste':>14}")

    for indice, (titulo, modelo, _) in enumerate(modelos):
        modelo.eval()
        with torch.no_grad():
            erro_treino = criterio_erro(modelo(X), Y).item()
            erro_validacao = None
            erro_teste = criterio_erro(modelo(X_plot), Y_plot).item()
        validacao_formatada = (
            f"{erro_validacao:>14.6f}"
            if erro_validacao is not None else f"{'-':>14}"
        )
        print(
            f"{titulo:<52} {erro_treino:>14.6f} "
            f"{validacao_formatada} {erro_teste:>14.6f}"
        )

    print("\nPesos e vieses (ganhos) das redes")
    for titulo, modelo, _ in modelos[:3]:
        print(f"\n{titulo}")
        numero_camada = 0
        soma_normas_pesos = 0.0
        for camada in modelo.modules():
            if isinstance(camada, nn.Linear):
                numero_camada += 1
                #print(f"  Camada linear {numero_camada}:")
                soma_normas_pesos += torch.linalg.matrix_norm(
                    camada.weight.detach(), ord=2
                ).item()**2
                #if camada.bias is not None:
                    #print(f"    Vieses/ganhos: {camada.bias.detach().cpu().numpy()}")
        print(f"  Soma das normas 2 dos pesos: {soma_normas_pesos:.6f}")

    for titulo, modelo, _ in modelos[3:4]:
        print(f"\n{titulo}")
        numero_camada = 0
        soma_norma_1 = 0.0
        for camada in modelo.modules():
            if isinstance(camada, nn.Linear):
                numero_camada += 1
                #print(f"  Camada linear {numero_camada}:")
                soma_norma_1 += camada.weight.abs().sum().item()
                #if camada.bias is not None:
                    #print(f"    Vieses/ganhos: {camada.bias.detach().cpu().numpy()}")
        print(f"  Soma das normas 1 dos pesos: {soma_norma_1:.6f}")

    for titulo, modelo, _ in modelos[4:5]:
        maiores_normas_colunas = [
            torch.linalg.vector_norm(camada.weight.detach(), ord=2, dim=0).max().item()
            for camada in modelo.modules()
            if isinstance(camada, nn.Linear)
        ]
        print(f"\n{titulo}")
        print(f"  Maior norma L2 de coluna: {max(maiores_normas_colunas):.6f}")

    # Calcula as predições uma vez e reutiliza nas três figuras
    predicoes = []
    for _, modelo, _ in modelos:
        modelo.eval()
        with torch.no_grad():
            y_predito = modelo(X_plot).numpy().ravel()
        predicoes.append(y_predito)

    figuras = [
        # 1) Dois modelos sem regularização
        plotar_modelos(
            [0, 1], "Modelos sem regularização", (11, 5),
            modelos, x, y, x_plot, y_original, predicoes,
        ),
        # 2) Os três primeiros modelos n=15: sem regularização, L1 e L2
        plotar_modelos(
            [1, 3, 2], "Modelos n=15: sem regularização, L1 e L2", (16, 5),
            modelos, x, y, x_plot, y_original, predicoes,
        ),
        # 3) Modelos n=15 com L1, L2 e projeção por coluna
        plotar_modelos(
            [3, 2, 4], "Modelos n=15 com regularização", (16, 5),
            modelos, x, y, x_plot, y_original, predicoes,
        ),
        # 4) Comparação com os modelos n=15 sem regularização, L2 e parada
        plotar_modelos(
            [1, 2, 5],
            "Modelos n=15: sem regularização, L2 e parada antecipada",
            (16, 5),
            modelos, x, y, x_plot, y_original, predicoes,
        ),
    ]

    
    pasta_graficos = Path(__file__).resolve().parent / "graficos"
    pasta_graficos.mkdir(exist_ok=True)
    nomes_arquivos = [
        "modelos_sem_regularizacao.pdf",
        "modelos_n15_l1_l2.pdf",
        "modelos_n15_regularizacao.pdf",
        "modelo_n15_parada_antecipada.pdf",
    ]
    for figura, nome_arquivo in zip(figuras, nomes_arquivos):
        figura.savefig(pasta_graficos / nome_arquivo)
    print(f"\nGráficos salvos em PDF em: {pasta_graficos}")
    plt.show()

if __name__ == "__main__":
    executar()