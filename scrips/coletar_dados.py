
import serial
import csv
import time
import re
from pathlib import Path
from datetime import datetime

# Configurações da coleta
PORTA = "COM4"       # Altere para a porta do seu ESP32
BAUDRATE = 115200
DURACAO = 3          # Duração de cada amostra em segundos

# Pasta onde os arquivos serão salvos
PASTA_DADOS = Path(__file__).resolve().parent.parent / "dados"
PASTA_DADOS.mkdir(parents=True, exist_ok=True)


def limpar_nome(texto):
    """Remove caracteres inválidos para nomes de arquivos."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', texto.strip())


def coletar_amostra(ser, gesto, pessoa, objeto):

    data = datetime.now().strftime("%Y%m%d_%H%M%S")

    nome_arquivo = (
        f"{limpar_nome(gesto)}_"
        f"{limpar_nome(pessoa)}_"
        f"{limpar_nome(objeto)}_"
        f"{data}.csv"
    )

    caminho = PASTA_DADOS / nome_arquivo

    linhas = []

    print("\nPrepare-se para realizar o gesto.")
    input("Pressione ENTER para iniciar a gravação...")

    # Descarta dados anteriores ao início da coleta
    ser.reset_input_buffer()

    print(f"\nGravando por {DURACAO} segundos...")

    inicio = time.monotonic()

    while time.monotonic() - inicio < DURACAO:

        linha = ser.readline().decode(
            "utf-8", errors="ignore"
        ).strip()

        if not linha:
            continue

        partes = linha.split(",")

        # Espera três valores: tempo, d1 e d2
        if len(partes) != 3:
            continue

        try:
            tempo, d1, d2 = partes

            tempo = int(tempo)
            d1 = float(d1)
            d2 = float(d2)

            linhas.append([tempo, d1, d2])

        except ValueError:
            continue

       # Salva os dados coletados
        with open(caminho, "w", newline="", encoding="utf-8") as arquivo:

            escritor = csv.writer(arquivo)

            escritor.writerow(
                ["tempo_ms", "t_rel_ms", "d1", "d2", "gesto", "pessoa", "objeto"]
            )

            t0 = linhas[0][0] if linhas else 0

            for tempo, d1, d2 in linhas:
                escritor.writerow(
                    [tempo, tempo - t0, d1, d2, gesto, pessoa, objeto]
                )


def main():

    print("=== SISTEMA DE COLETA DE DADOS ===")

    gesto = input("Nome do gesto: ")
    pessoa = input("Identificação da pessoa: ")
    objeto = input("Objeto utilizado: ")

    try:
        with serial.Serial(
            PORTA,
            BAUDRATE,
            timeout=1
        ) as ser:

            # Aguarda a inicialização da comunicação serial
            time.sleep(2)
            ser.reset_input_buffer()

            while True:

                coletar_amostra(
                    ser, gesto, pessoa, objeto
                )

                continuar = input(
                    "\nGravar outra amostra? (s/n): "
                ).lower()

                if continuar != "s":
                    break

    except serial.SerialException as erro:
        print(f"Erro na comunicação serial: {erro}")


if __name__ == "__main__":
    main()
