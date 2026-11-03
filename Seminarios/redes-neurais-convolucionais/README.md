# Seminário: Redes Neurais Convolucionais

CPE727 - Aprendizado Profundo, PEE/COPPE/UFRJ
Eraldo Silva Junior, 29/10/2026

```
redes-neurais-convolucionais/
├── README.md
├── slides.pdf          slides do seminário
├── referencias.bib     referências usadas nos slides
├── main.tex            fonte dos slides (Beamer)
├── figuras/            figuras usadas no main.tex
└── exemplos/
    ├── codigos.ipynb   CNN simples em PyTorch no MNIST
    ├── libs.py         funções do notebook (dados, treino, acurácia)
    └── DockerFile      ambiente para rodar o notebook
```

## Slides

Usam o tema metropolis. Compila com pdfLaTeX ou XeLaTeX, e no Overleaf funciona com o compilador padrão.

```bash
pdflatex main.tex
pdflatex main.tex
```

A lista de referências está escrita direto no `main.tex`. O `referencias.bib` tem as mesmas referências em BibTeX.

A marca ▶ NN_nome no canto dos slides indica a animação que foi passada naquele ponto da apresentação. No PDF ela é um link para `videos/NN_nome.mp4`, mas os vídeos não estão aqui, então esses links não abrem.

## Exemplos

O notebook já está com as saídas, dá para ler direto no GitHub. Ele faz uma convolução na mão com o filtro de Sobel, treina uma CNN com duas camadas convolucionais no MNIST (98% no teste em 3 épocas), mostra os kernels e mapas de características aprendidos e testa o que acontece quando a imagem é deslocada.


```bash
cd exemplos
docker build -f DockerFile -t cnn-exemplos .
docker run --rm -p 8888:8888 cnn-exemplos
```

e abrir o link `http://127.0.0.1:8888/...` que aparece no terminal.

Sem Docker (recomendo usar venv):

```bash
cd exemplos
pip install torch torchvision matplotlib numpy jupyter
jupyter notebook codigos.ipynb
```

O MNIST é baixado na pasta `dados/` na primeira execução.

## Referência principal

Goodfellow, Bengio e Courville. Deep Learning, cap. 9 (Convolutional Networks). MIT Press, 2016.
https://www.deeplearningbook.org/contents/convnets.html
