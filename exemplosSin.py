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

# Reprodutibilidade
np.random.seed(42)
torch.manual_seed(42)

# Dataset: pontos igualmente distribuídos entre 0 e 2*pi
x = np.linspace(0, 2 * np.pi, N, dtype=np.float32)
y = np.sin(x) + np.random.normal(0, sigma, N).astype(np.float32)

X = torch.from_numpy(x).reshape(-1, 1)
Y = torch.from_numpy(y).reshape(-1, 1)


def criar_modelo(n):
    camadas = []
    tamanho_entrada = 1
    ativacao = nn.Tanh()

    for _ in range(depth):
        camadas.extend([nn.Linear(tamanho_entrada, n), ativacao])
        tamanho_entrada = n

    camadas.append(nn.Linear(tamanho_entrada, 1))
    return nn.Sequential(*camadas)


def treinar_modelo(n, usar_l2=False, usar_l1=False, usar_projecao=False):
    # A mesma semente permite comparar os modelos de 15 neurônios
    # com a mesma inicialização.
    torch.manual_seed(123)

    modelo = criar_modelo(n)
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

    regularizacao = (
        "L1" if usar_l1 else "L2" if usar_l2
        else "projeção L2 por coluna" if usar_projecao else "não"
    )
    nome_modelo = f"n={n}, regularização {regularizacao}"
    largura_barra = 30
    intervalo_atualizacao = max(1, epochs // 100)

    for epoca in range(epochs):
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

        if (epoca + 1) % intervalo_atualizacao == 0 or epoca + 1 == epochs:
            progresso = (epoca + 1) / epochs
            preenchido = int(largura_barra * progresso)
            barra = "=" * preenchido + " " * (largura_barra - preenchido)
            print(
                f"\rTreinando {nome_modelo}: [{barra}] "
                f"{progresso:6.1%} | perda: {perda.item():.6f}",
                end="\n" if epoca + 1 == epochs else "",
                flush=True,
            )

    return modelo


# Treina os cinco modelos
configuracoes = [
    (f"n=1, depth={depth}, epochs={epochs}", 1, False, False, False),
    (f"n=15, depth={depth}, epochs={epochs}", 15, False, False, False),
    (f"n=15, depth={depth}, epochs={epochs}, regularização L2", 15, True, False, False),
    (f"n=15, depth={depth}, epochs={epochs}, regularização L1", 15, False, True, False),
    (f"n=15, depth={depth}, projeção L2 por coluna", 15, False, False, True),
]

modelos = [
    (titulo, treinar_modelo(n, usar_l2, usar_l1, usar_projecao))
    for titulo, n, usar_l2, usar_l1, usar_projecao in configuracoes
]

# Predições em uma grade densa
x_plot = np.linspace(0, 2 * np.pi, 500, dtype=np.float32)
X_plot = torch.from_numpy(x_plot).reshape(-1, 1)
y_original = np.sin(x_plot)
Y_plot = torch.from_numpy(y_original).reshape(-1, 1)

# Erros MSE dos quatro modelos
criterio_erro = nn.MSELoss()
print("\nErros dos modelos (MSE)")
print(f"{'Modelo':<40} {'Treinamento':>14} {'Teste':>14}")

for titulo, modelo in modelos:
    modelo.eval()
    with torch.no_grad():
        erro_treino = criterio_erro(modelo(X), Y).item()
        erro_teste = criterio_erro(modelo(X_plot), Y_plot).item()
    print(f"{titulo:<40} {erro_treino:>14.6f} {erro_teste:>14.6f}")

print("\nPesos e vieses (ganhos) das redes")
for titulo, modelo in modelos[:3]:
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

for titulo, modelo in modelos[3:4]:
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

for titulo, modelo in modelos[4:]:
    maiores_normas_colunas = [
        torch.linalg.vector_norm(camada.weight.detach(), ord=2, dim=0).max().item()
        for camada in modelo.modules()
        if isinstance(camada, nn.Linear)
    ]
    print(f"\n{titulo}")
    print(f"  Maior norma L2 de coluna: {max(maiores_normas_colunas):.6f}")


# Calcula as predições uma vez e reutiliza nas três figuras
predicoes = []
for _, modelo in modelos:
    modelo.eval()
    with torch.no_grad():
        y_predito = modelo(X_plot).numpy().ravel()
    predicoes.append(y_predito)

def plotar_modelos(indices, titulo_figura, figsize):
    fig, eixos = plt.subplots(1, len(indices), figsize=figsize, sharey=True)
    for eixo, indice in zip(eixos, indices):
        titulo, _ = modelos[indice]
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

# 1) Dois modelos sem regularização
plotar_modelos([0, 1], "Modelos sem regularização", (11, 5))
# 2) Os três primeiros modelos n=15: sem regularização, L1 e L2
plotar_modelos([1, 3, 2], "Modelos n=15: sem regularização, L1 e L2", (16, 5))
# 3) Modelos n=15 com L1, L2 e projeção por coluna
plotar_modelos([3, 2, 4], "Modelos n=15 com regularização", (16, 5))

plt.show()