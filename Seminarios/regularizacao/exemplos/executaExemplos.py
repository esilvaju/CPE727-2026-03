from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import Seminarios.regularizacao.exemplos.exemplos_sin as exemplos_sin
import Seminarios.regularizacao.exemplos.exemplo_multitask as exemplo_multitask
import Seminarios.regularizacao.exemplos.exemplos_MNIST as exemplos_MNIST


def executar():
	print("\n=== Exemplo 1: predição com modelo gerador sin(x) ===")
	exemplos_sin.executar()

	print("\n=== Exemplo 2: aprendizado multitarefas ===")
	exemplo_multitask.executar()

	print("\n=== Exemplo 3: classificação com dataset MNIST ===")
	exemplos_MNIST.executar()


if __name__ == "__main__":
	executar()