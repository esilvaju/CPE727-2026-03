"""Funções auxiliares do notebook codigos.ipynb (dados, treino e avaliação)."""
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms


def escolher_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def carregar_mnist(pasta="dados", batch_size=128):
    """Baixa o MNIST (na primeira vez) e devolve os datasets e os DataLoaders de treino e teste."""
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    treino = datasets.MNIST(pasta, train=True, download=True, transform=tf)
    teste = datasets.MNIST(pasta, train=False, download=True, transform=tf)
    dl_treino = torch.utils.data.DataLoader(treino, batch_size=batch_size, shuffle=True)
    dl_teste = torch.utils.data.DataLoader(teste, batch_size=1000)
    return treino, teste, dl_treino, dl_teste


def acuracia(modelo, dl, device, deslocamento=0):
    """Fração de acertos em dl. Com deslocamento > 0, as imagens andam esse número de pixels para a direita."""
    modelo.eval()
    certos = 0
    with torch.no_grad():
        for x, y in dl:
            if deslocamento:
                x = torch.roll(x, shifts=deslocamento, dims=3)
            certos += (modelo(x.to(device)).argmax(1) == y.to(device)).sum().item()
    return certos / len(dl.dataset)


def treinar(modelo, dl_treino, dl_teste, device, epocas=3, lr=1e-3):
    otim = torch.optim.Adam(modelo.parameters(), lr=lr)
    for epoca in range(epocas):
        modelo.train()
        for x, y in dl_treino:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(modelo(x), y)
            otim.zero_grad()
            loss.backward()
            otim.step()
        print(f"época {epoca + 1}: acurácia no teste = {acuracia(modelo, dl_teste, device):.2%}")
