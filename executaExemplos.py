import exemplos_sin
import exemplo_multitask
import exemplos_MNIST


def executar():
	print("\n=== Exemplo 1: predição com modelo gerador sin(x) ===")
	exemplos_sin.executar()

	print("\n=== Exemplo 2: aprendizado multitarefas ===")
	exemplo_multitask.executar()

	print("\n=== Exemplo 3: classificação com dataset MNIST ===")
	exemplos_MNIST.executar()


if __name__ == "__main__":
	executar()