import hashlib
import json
from pathlib import Path
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.dataloaders.MNISTLoader.mnist_loader import (
    adicionar_ruido_gaussiano,
    carregar_dados,
    criar_dataloader,
)

import torch
from torch import nn


N = 64
PROFUNDIDADE = 5
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
SEMENTE = 42
DESVIO_PADRAO_RUIDO = 0.1
COPIAS_RUIDO_TESTE = 2
DESVIO_PADRAO_RUIDO_PESOS = 0.02
PROBABILIDADE_DROPOUT = 0.2

def criar_modelo(n, profundidade=PROFUNDIDADE, probabilidade_dropout=0.0):
    camadas = []
    tamanho_entrada = 28 * 28

    for _ in range(profundidade):
        camadas.extend([nn.Linear(tamanho_entrada, n), nn.ReLU()])
        if probabilidade_dropout > 0:
            camadas.append(nn.Dropout(probabilidade_dropout))
        tamanho_entrada = n

    camadas.append(nn.Linear(tamanho_entrada, 10))
    return nn.Sequential(*camadas)


def prever_com_ruido_nos_pesos(modelo, entradas, sigma):
    parametros = {
        nome: parametro + torch.randn_like(parametro) * sigma
        if parametro.ndim > 1
        else parametro
        for nome, parametro in modelo.named_parameters()
    }
    return torch.func.functional_call(modelo, parametros, (entradas,))


def avaliar_modelo(modelo, carregador, criterio, dispositivo):
    modelo.eval()
    perda_total = 0.0
    acertos = 0
    total = 0

    with torch.no_grad():
        for entradas, rotulos in carregador:
            entradas = entradas.to(dispositivo)
            rotulos = rotulos.to(dispositivo)
            logits = modelo(entradas)
            perda_total += criterio(logits, rotulos).item()
            acertos += (logits.argmax(dim=1) == rotulos).sum().item()
            total += rotulos.size(0)

    return perda_total / total, acertos / total


def caminho_modelo_salvo(configuracao):
    callbacks = {
        chave: configuracao.get(chave)
        for chave in ("penalidade", "apos_passo")
        if configuracao.get(chave) is not None
    }
    if callbacks and not configuracao.get("cache_id"):
        raise ValueError(
            "Defina 'cache_id' na configuração para identificar a versão "
            "dos callbacks de regularização."
        )

    identidade = {
        chave: valor
        for chave, valor in configuracao.items()
        if not callable(valor) and chave.casefold() not in {"epoch", "epochs"}
    }
    identidade["callbacks"] = {
        chave: f"{callback.__module__}.{callback.__qualname__}"
        for chave, callback in callbacks.items()
    }
    identidade["treinamento"] = {
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
    }
    identidade_json = json.dumps(identidade, sort_keys=True, default=str)
    identificador = hashlib.sha256(identidade_json.encode("utf-8")).hexdigest()[:12]
    nome = re.sub(r"[^a-zA-Z0-9_-]+", "_", configuracao["nome"]).strip("_")
    profundidade = configuracao.get("profundidade", PROFUNDIDADE)
    pasta_modelos = Path(__file__).resolve().parent / "modelos_treinados"
    nome_arquivo = (
        f"{nome}_N{configuracao['n']}_lr{LEARNING_RATE:g}_"
        f"prof{profundidade}_batch{BATCH_SIZE}_{identificador}.pt"
    )
    return pasta_modelos / nome_arquivo, identidade


def carregar_modelo_salvo(configuracao):
    caminho, identidade = caminho_modelo_salvo(configuracao)
    if not caminho.is_file():
        return None

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(caminho, map_location=dispositivo, weights_only=True)
    if checkpoint["identidade"] != identidade:
        return None

    modelo = criar_modelo(
        configuracao["n"],
        configuracao.get("profundidade", PROFUNDIDADE),
        configuracao.get("probabilidade_dropout", 0.0),
    ).to(dispositivo)
    modelo.load_state_dict(checkpoint["estado_modelo"])
    modelo.eval()
    print(f"Modelo carregado: {configuracao['nome']} ({caminho.name})", flush=True)
    return modelo, checkpoint["resultado"]


def salvar_modelo(configuracao, modelo, resultado):
    caminho, identidade = caminho_modelo_salvo(configuracao)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho_temporario = caminho.with_suffix(".tmp")
    torch.save(
        {
            "identidade": identidade,
            "estado_modelo": modelo.state_dict(),
            "resultado": resultado,
        },
        caminho_temporario,
    )
    caminho_temporario.replace(caminho)
    print(f"Modelo salvo: {caminho}", flush=True)


def treinar_modelo(
    configuracao, carregador_treino, carregador_treino_avaliacao, carregador_teste
):
    torch.manual_seed(SEMENTE)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = criar_modelo(
        configuracao["n"],
        configuracao.get("profundidade", PROFUNDIDADE),
        configuracao.get("probabilidade_dropout", 0.0),
    ).to(dispositivo)

    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.Adam(
        modelo.parameters(),
        lr=LEARNING_RATE,
        weight_decay=configuracao.get("weight_decay", 0.0),
    )
    penalidade = configuracao.get("penalidade")
    apos_passo = configuracao.get("apos_passo")
    ruido_pesos_sigma = configuracao.get("ruido_pesos_sigma")
    largura_barra = 30
    intervalo_atualizacao = max(1, EPOCHS // 100)

    for epoch in range(EPOCHS):
        modelo.train()
        perda_total = 0.0
        total = 0

        for entradas, rotulos in carregador_treino:
            entradas = entradas.to(dispositivo)
            rotulos = rotulos.to(dispositivo)

            otimizador.zero_grad()
            if ruido_pesos_sigma is None:
                logits = modelo(entradas)
            else:
                logits = prever_com_ruido_nos_pesos(
                    modelo, entradas, ruido_pesos_sigma
                )
            perda = criterio(logits, rotulos)
            if penalidade is not None:
                perda = perda + penalidade(modelo)
            perda.backward()
            otimizador.step()

            if apos_passo is not None:
                with torch.no_grad():
                    apos_passo(modelo)

            tamanho_lote = rotulos.size(0)
            perda_total += perda.item() * tamanho_lote
            total += tamanho_lote

        epoch_atual = epoch + 1
        if epoch_atual % intervalo_atualizacao == 0 or epoch_atual == EPOCHS:
            progresso = epoch_atual / EPOCHS
            preenchido = int(largura_barra * progresso)
            barra = "=" * preenchido + " " * (largura_barra - preenchido)
            print(
                f"\rTreinando {configuracao['nome']}: [{barra}] "
                f"{progresso:6.1%} | perda: {perda_total / total:.6f}",
                end="\n" if epoch_atual == EPOCHS else "",
                flush=True,
            )

    perda_treino, acuracia_treino = avaliar_modelo(
        modelo, carregador_treino_avaliacao, criterio, dispositivo
    )
    perda_teste, acuracia_teste = avaliar_modelo(
        modelo, carregador_teste, criterio, dispositivo
    )
    resultado = {
        "nome": configuracao["nome"],
        "perda_treino": perda_treino,
        "acuracia_treino": acuracia_treino,
        "perda_teste": perda_teste,
        "acuracia_teste": acuracia_teste,
    }
    return modelo, resultado


def plotar_amostras(modelos, dados_teste, indices):
    imagens, rotulos = dados_teste.tensors
    rng = np.random.default_rng()
    indices_atuais = np.asarray(indices).copy()
    figuras = []

    for modelo, resultado in modelos:
        fig, eixos = plt.subplots(8, 12, figsize=(12.8, 6.4))
        fig.suptitle(f"Amostras de teste: {resultado['nome']}")
        artistas = []
        for eixo in eixos.flat:
            imagem_plotada = eixo.imshow(
                np.zeros((28, 28)), cmap="gray", vmin=0.0, vmax=1.0
            )
            titulo = eixo.text(
                0.5,
                0.02,
                "",
                transform=eixo.transAxes,
                ha="center",
                va="bottom",
                fontsize=5.5,
                color="black",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                    "pad": 0.4,
                },
            )
            borda = Rectangle(
                (0, 0),
                1,
                1,
                transform=eixo.transAxes,
                fill=False,
                edgecolor="green",
                linewidth=3,
                clip_on=False,
            )
            eixo.add_patch(borda)
            eixo.axis("off")
            artistas.append((imagem_plotada, titulo, borda))

        fig.tight_layout(rect=(0, 0.07, 1, 0.95))
        eixo_botao = fig.add_axes((0.41, 0.015, 0.18, 0.04))
        botao = Button(eixo_botao, "Sortear outras")
        fig._botao_sorteio = botao
        figuras.append((modelo, fig, artistas, botao))

    def atualizar_figuras(indices_novos):
        indices_tensor = torch.as_tensor(indices_novos, dtype=torch.long)
        imagens_selecionadas = imagens[indices_tensor]
        rotulos_selecionados = rotulos[indices_tensor]

        for modelo, fig, artistas, _ in figuras:
            dispositivo = next(modelo.parameters()).device
            modelo.eval()
            with torch.no_grad():
                predicoes = modelo(
                    imagens_selecionadas.to(dispositivo)
                ).argmax(dim=1).cpu()

            for (imagem_plotada, titulo, borda), imagem, rotulo, predicao in zip(
                artistas, imagens_selecionadas, rotulos_selecionados, predicoes
            ):
                rotulo_real = rotulo.item()
                rotulo_predito = predicao.item()
                imagem_plotada.set_data(imagem.reshape(28, 28).numpy())
                titulo.set_text(
                    f"Real: {rotulo_real} | Predição: {rotulo_predito}"
                )
                borda.set_edgecolor(
                    "green" if rotulo_real == rotulo_predito else "red"
                )

            fig.canvas.draw_idle()

    def sortear_outras_amostras(event):
        indices_novos = rng.choice(
            len(imagens), size=len(indices_atuais), replace=False
        )
        while np.array_equal(indices_novos, indices_atuais):
            indices_novos = rng.choice(
                len(imagens), size=len(indices_atuais), replace=False
            )
        indices_atuais[:] = indices_novos
        atualizar_figuras(indices_atuais)

    for _, _, _, botao in figuras:
        botao.on_clicked(sortear_outras_amostras)

    atualizar_figuras(indices_atuais)
    return [figura for _, figura, _, _ in figuras]


def plotar_comparacao(resultados):
    nomes = [resultado["nome"] for resultado in resultados]
    posicoes = np.arange(len(nomes))
    largura = 0.36
    fig, eixo = plt.subplots(figsize=(7, 4))

    eixo.bar(
        posicoes - largura / 2,
        [resultado["acuracia_treino"] * 100 for resultado in resultados],
        largura,
        label="Treino",
    )
    eixo.bar(
        posicoes + largura / 2,
        [resultado["acuracia_teste"] * 100 for resultado in resultados],
        largura,
        label="Teste",
    )
    eixo.set_title("Acurácia (%)")
    eixo.legend(loc="lower right")
    eixo.set_ylim(90, 100)
    eixo.set_xticks(posicoes, nomes, rotation=30, ha="right")
    eixo.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def executar():
    torch.manual_seed(SEMENTE)
    _, _, dados_treino, dados_teste = carregar_dados(
        Path(__file__).resolve().parent, BATCH_SIZE
    )
    rng = np.random.default_rng(SEMENTE)
    indices_amostras = rng.choice(len(dados_teste), size=96, replace=False)

    configuracoes = [
        {
            "nome": "Modelo injeção de ruído - entradas",
            "n": N,
            "profundidade": PROFUNDIDADE,
            "weight_decay": 0.0,
            "penalidade": None,
            "apos_passo": None,
            "ruido_sigma": DESVIO_PADRAO_RUIDO,
            "copias_ruido_teste": COPIAS_RUIDO_TESTE,
        },
        {
            "nome": "Modelo injeção de ruído - pesos",
            "n": N,
            "profundidade": PROFUNDIDADE,
            "weight_decay": 0.0,
            "penalidade": None,
            "apos_passo": None,
            "ruido_pesos_sigma": DESVIO_PADRAO_RUIDO_PESOS,
        },
        {
            "nome": "Modelo regularização dropout",
            "n": N,
            "profundidade": PROFUNDIDADE,
            "weight_decay": 0.0,
            "penalidade": None,
            "apos_passo": None,
            "probabilidade_dropout": PROBABILIDADE_DROPOUT,
        },
    ]

    resultados = []
    modelos_treinados = []
    for configuracao in configuracoes:
        dados_treino_modelo = dados_treino
        dados_teste_modelo = dados_teste
        if configuracao.get("ruido_sigma") is not None:
            sigma = configuracao["ruido_sigma"]
            dados_treino_modelo = adicionar_ruido_gaussiano(
                dados_treino, quantidade_copias=1, seed=SEMENTE, sigma=sigma
            )
            dados_teste_modelo = adicionar_ruido_gaussiano(
                dados_teste,
                quantidade_copias=configuracao["copias_ruido_teste"],
                seed=SEMENTE + 1,
                sigma=sigma,
                incluir_originais=True,
            )

        modelo_salvo = carregar_modelo_salvo(configuracao)
        if modelo_salvo is None:
            carregador_treino_modelo = criar_dataloader(
                dados_treino_modelo, BATCH_SIZE, shuffle=True, seed=SEMENTE
            )
            carregador_treino_avaliacao = criar_dataloader(
                dados_treino_modelo, 512
            )
            carregador_teste_modelo = criar_dataloader(dados_teste_modelo, 512)
            modelo, resultado = treinar_modelo(
                configuracao,
                carregador_treino_modelo,
                carregador_treino_avaliacao,
                carregador_teste_modelo,
            )
            salvar_modelo(configuracao, modelo, resultado)
        else:
            modelo, resultado = modelo_salvo
        modelos_treinados.append((modelo, resultado))
        resultados.append(resultado)

    print("\nComparação dos modelos")
    print(f"{'Modelo':<40} {'Acurácia treino':>16} {'Acurácia teste':>15}")
    for resultado in resultados:
        print(
            f"{resultado['nome']:<40} "
            f"{resultado['acuracia_treino'] * 100:>15.2f}% "
            f"{resultado['acuracia_teste'] * 100:>14.2f}%"
        )

    figuras_amostras = plotar_amostras(
        modelos_treinados, dados_teste, indices_amostras
    )
    figura_comparacao = plotar_comparacao(resultados)

    pasta_graficos = Path(__file__).resolve().parent / "graficos"
    pasta_graficos.mkdir(exist_ok=True)
    nomes_amostras = [
        "mnist_amostras_ruido_entradas.pdf",
        "mnist_amostras_ruido_pesos.pdf",
        "mnist_amostras_dropout.pdf",
    ]
    for figura, nome_arquivo in zip(figuras_amostras, nomes_amostras):
        figura.savefig(pasta_graficos / nome_arquivo)
    figura_comparacao.savefig(pasta_graficos / "mnist_comparacao_acuracia.pdf")
    print(f"\nGráficos do MNIST salvos em PDF em: {pasta_graficos}")
    plt.show()


if __name__ == "__main__":
    executar()




