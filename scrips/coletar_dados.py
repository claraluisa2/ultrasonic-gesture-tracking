import csv
import re
import time
from datetime import datetime
from pathlib import Path

import serial

# Configurações da coleta
PORTA = "COM4"       # Altere para a porta do seu ESP32
BAUDRATE = 115200
DURACAO = 3          # Duração de cada amostra em segundos

# Pasta onde os arquivos serão salvos
PASTA_DADOS = Path(__file__).resolve().parent.parent / "dados"
PASTA_DADOS.mkdir(parents=True, exist_ok=True)


def limpar_nome(texto):
    """Remove caracteres inválidos para nomes de arquivos."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", texto.strip())


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

        linha = ser.readline().decode("utf-8", errors="ignore").strip()

        if not linha:
            continue

        partes = linha.split(",")

        # Espera três valores: tempo, d1 e d2
        if len(partes) != 3:
            continue

        try:
            tempo, d1, d2 = partes
            linhas.append([int(tempo), float(d1), float(d2)])
        except ValueError:
            continue

    # ---- O salvamento fica FORA do while ----
    # (antes ele estava dentro do loop e reescrevia o arquivo inteiro a cada
    #  leitura, atrasando a leitura da serial)
    if not linhas:
        print("Nenhum dado recebido. Confira a porta e se o ESP32 está enviando.")
        return

    t0 = linhas[0][0]

    with open(caminho, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(
            ["tempo_ms", "t_rel_ms", "d1", "d2", "gesto", "pessoa", "objeto"]
        )
        for tempo, d1, d2 in linhas:
            escritor.writerow([tempo, tempo - t0, d1, d2, gesto, pessoa, objeto])

    # Os dados ficam crus (-1 = falha do sensor); a limpeza é feita depois.
    falhas = sum(1 for _, d1, d2 in linhas if d1 < 0 or d2 < 0)
    print(
        f"Salvo: {nome_arquivo} | {len(linhas)} leituras | "
        f"{100 * falhas / len(linhas):.0f}% com alguma falha"
    )


def main():

    print("=== SISTEMA DE COLETA DE DADOS ===")

    gesto = input("Nome do gesto: ")
    pessoa = input("Identificação da pessoa: ")
    objeto = input("Objeto utilizado: ")

    try:
        with serial.Serial(PORTA, BAUDRATE, timeout=1) as ser:

            # Aguarda a inicialização da comunicação serial
            time.sleep(2)
            ser.reset_input_buffer()

            while True:
                coletar_amostra(ser, gesto, pessoa, objeto)

                continuar = input("\nGravar outra amostra? (s/n): ").lower()
                if continuar != "s":
                    break

    except serial.SerialException as erro:
        print(f"Erro na comunicação serial: {erro}")


if __name__ == "__main__":
    main()