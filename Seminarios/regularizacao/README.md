# Exemplos de Regularização

Este diretório contém três exemplos em PyTorch sobre regularização aplicados ao aprendizado profundo. Todos os exemplos utilizam `DataLoader`s específicos, localizados em `src/dataloaders`.

## Requisitos

- Docker Desktop em execução;
- Docker Compose não é necessário;
- Os arquivos IDX do MNIST devem permanecer neste diretório:
  - `train-images.idx3-ubyte`;
  - `train-labels.idx1-ubyte`;
  - `t10k-images.idx3-ubyte`;
  - `t10k-labels.idx1-ubyte`.

## Execução com Docker

Os comandos devem ser executados a partir da raiz do repositório, isto é, do diretório `CPE727-2026-03`.

### Construir a imagem

```powershell
docker build -f Seminarios/regularizacao/exemplos/Dockerfile -t cpe727-regularizacao .
```

O ponto final indica a raiz do repositório como contexto de build. Isso permite que a imagem copie tanto os exemplos quanto os pacotes em `src/dataloaders`.

### Executar os exemplos

```powershell
docker run --rm -it cpe727-regularizacao
```

O comando executa `executaExemplos.py`, que roda os três exemplos em sequência. A imagem usa o backend `Agg` do Matplotlib, adequado para execução sem interface gráfica.

### Persistir resultados no host

Para manter os PDFs, checkpoints e modelos fora do contêiner, monte os diretórios de saída:

```powershell
docker run --rm -it `
  -v "${PWD}\Seminarios\regularizacao\exemplos\graficos:/app/Seminarios/regularizacao/exemplos/graficos" `
  -v "${PWD}\Seminarios\regularizacao\exemplos\modelos_treinados:/app/Seminarios/regularizacao/exemplos/modelos_treinados" `
  cpe727-regularizacao
```

Os gráficos são salvos em `graficos/` e os modelos do MNIST em `modelos_treinados/`.

## Exemplos

### 1. Predição de `sin(x)`

Arquivo: `exemplos_sin.py`

O `SinLoader` cria um dataset sintético com 20 pontos uniformemente distribuídos no intervalo `[0, 2*pi]`. Cada alvo é calculado como:

```text
y = sin(x) + ruido gaussiano
```

O ruído tem desvio padrão `0.3`. Uma rede totalmente conectada com cinco camadas ocultas `Tanh` é treinada em seis configurações:

1. uma rede com um neurônio por camada, sem regularização;
2. uma rede com 15 neurônios por camada, sem regularização;
3. regularização L2 usando `weight_decay` no otimizador;
4. regularização L1 adicionada à função de perda;
5. projeção L2 por coluna dos pesos, limitando a norma máxima;
6. parada antecipada baseada em um conjunto de validação.

O último caso estima o melhor número de épocas, restaura os melhores pesos e depois retreina o modelo com todo o conjunto. Os resultados e curvas são salvos em arquivos PDF no diretório `graficos/`.

### 2. Aprendizado multitarefa

Arquivo: `exemplo_multitask.py`

O `MultitaskLoader` gera 2.000 amostras tabulares sintéticas com as seguintes características:

- idade;
- índice de massa corporal;
- pressão sistólica;
- horas de exercício;
- histórico familiar de diabetes.

O mesmo conjunto alimenta duas tarefas:

- regressão da concentração de glicose;
- classificação binária de diabetes.

A rede possui um tronco compartilhado com três camadas `Linear` de 15 neurônios e duas cabeças específicas: uma para regressão e outra para classificação. A perda total combina MSE da glicose e entropia cruzada binária da diabetes.

Os dados são padronizados e divididos em treino, validação e teste nas proporções 70%, 15% e 15%. Neste exemplo não há regularização explícita: `weight_decay` é zero. O foco é demonstrar o compartilhamento de representações entre as duas tarefas e a avaliação conjunta por RMSE, MAE, acurácia, precisão, revocação, F1 e ROC-AUC.

### 3. Classificação com MNIST

Arquivo: `exemplos_MNIST.py`

O `MNISTLoader` lê os arquivos IDX locais do MNIST, normaliza os pixels para `[0, 1]` e cria os `DataLoader`s de treino e teste. O modelo é um perceptron multicamadas com cinco camadas ocultas de 64 neurônios, ativação `ReLU` e uma camada de saída com dez classes.

São comparadas três formas de regularização:

1. **Ruído nas entradas:** adiciona ruído gaussiano com desvio padrão `0.1` às imagens durante o treinamento. No teste, avalia as imagens originais e duas cópias ruidosas;
2. **Ruído nos pesos:** injeta ruído gaussiano com desvio padrão `0.02` nos pesos durante o cálculo da predição. Os vieses não recebem ruído;
3. **Dropout:** aplica `Dropout(p=0.2)` após cada camada oculta.

Os modelos treinados são armazenados em `modelos_treinados/` e reutilizados quando a configuração permanece a mesma. As comparações de acurácia e as amostras classificadas são salvas em `graficos/`.

## Estrutura

```text
exemplos/
├── Dockerfile
├── executaExemplos.py
├── exemplos_sin.py
├── exemplo_multitask.py
├── exemplos_MNIST.py
├── requirements.txt
├── graficos/
├── modelos_treinados/
└── *.idx*-ubyte

../../../src/dataloaders/
├── SinLoader/
├── MultitaskLoader/
├── MNISTLoader/
└── TemplateLoader/
```

O `TemplateLoader` é mantido como referência para a criação de novos loaders.

## Execução local sem Docker

Com Python 3.10 ou superior e as dependências instaladas:

```powershell
pip install -r Seminarios/regularizacao/exemplos/requirements.txt
python Seminarios/regularizacao/exemplos/executaExemplos.py
```

## Licença

Consulte o arquivo `LICENSE.txt` na raiz do repositório.
