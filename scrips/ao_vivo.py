
"""
Monitoramento em tempo real + coleta guiada dos sensores HC-SR04.

Serial esperada:
    tempo_ms,d1,d2

Controles:
    R = inicia uma coleta
    C = limpa o histórico visual
    S = mostra estatísticas das leituras brutas
    Q = fecha a janela

A coleta é realizada sem interromper o monitoramento.
Os dados brutos são preservados, inclusive -1 (sensor inválido).
"""

import collections
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import serial

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.widgets import TextBox


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

PORTA = "COM4"            # No Linux, provavelmente será /dev/ttyUSB0
BAUDRATE = 115200

JANELA_S = 4.0           # Histórico visual em segundos
LIM_CM = (0, 100)        # Limites dos gráficos
AREA_UTIL = (40, 60)      # Área de referência do gesto

ALFA = 0.5              # Suavização exponencial

DURACAO_COLETA = 3.0    # Duração de cada gesto
CONTAGEM_REGRESSIVA = 3  # Tempo para se preparar antes da coleta

# Número máximo de pontos mantidos nos gráficos.
# Os dados salvos em CSV não são limitados por este valor.
MAX_PONTOS_GRAFICO = 200

# Pasta "dados" localizada dois níveis acima deste script.
PASTA_DADOS = Path(__file__).resolve().parent.parent / "dados"


# ============================================================
# 2. FILTRO DOS SENSORES
# ============================================================

class FiltroCausal:
    """
    Filtro composto por:
    1. Mediana das últimas 3 leituras válidas.
    2. Média exponencial.

    Ele é utilizado somente para visualização.

    A coleta salva os dados brutos, para que possamos decidir
    o pré-processamento posteriormente.
    """

    def __init__(self, alfa):
        self.alfa = alfa
        self.reiniciar()

    def reiniciar(self):
        self.recentes = collections.deque(maxlen=3)
        self.y = np.nan
        self.falhas_seguidas = 0

    def passo(self, x):
        # Leitura inválida: não inventamos uma distância.
        if np.isnan(x):
            self.falhas_seguidas += 1

            # Após muitas falhas, descartamos o estado anterior.
            if self.falhas_seguidas >= 5:
                self.reiniciar()
                self.falhas_seguidas = 5

            return np.nan

        self.falhas_seguidas = 0
        self.recentes.append(x)

        # A mediana reduz o efeito de picos isolados.
        mediana = float(np.median(self.recentes))

        # A média exponencial suaviza as oscilações.
        if np.isnan(self.y):
            self.y = mediana
        else:
            self.y = (
                self.alfa * mediana
                + (1 - self.alfa) * self.y
            )

        return self.y


# ============================================================
# 3. INTERPRETAÇÃO DA SERIAL
# ============================================================

def interpretar(linha):
    """
    Converte uma linha serial:
        tempo_ms,d1,d2

    Retorna:
        (tempo, d1, d2)

    Para os gráficos, -1 vira NaN.
    O valor bruto original continua disponível para ser salvo.
    """

    partes = linha.strip().split(",")

    if len(partes) != 3:
        return None

    try:
        tempo = int(partes[0])
        d1 = float(partes[1])
        d2 = float(partes[2])
    except ValueError:
        return None

    # Valores brutos são mantidos separadamente.
    d1_bruto = d1
    d2_bruto = d2

    # NaN é utilizado internamente para o gráfico.
    d1_plot = d1 if d1 > 0 else np.nan
    d2_plot = d2 if d2 > 0 else np.nan

    return tempo, d1_bruto, d2_bruto, d1_plot, d2_plot


class DetectorMovimento:

    def __init__(self):
        # Estado atual do detector
        self.em_movimento = False

        # Guardará as leituras anteriores
        self.d1_anterior = None
        self.d2_anterior = None

        # Indicador de movimento
        self.movimento = 0

        # Futuramente: contadores para confirmar
        # o início e o fim do gesto
        self.contador_inicio = 0
        self.contador_fim = 0


# ============================================================
# 4. PAINEL PRINCIPAL
# ============================================================

class Painel:

    def __init__(self, ser):

        self.ser = ser

        # Histórico dos gráficos.
        self.t = collections.deque()
        self.r1 = collections.deque()
        self.r2 = collections.deque()
        self.f1 = collections.deque()
        self.f2 = collections.deque()

        self.filtro1 = FiltroCausal(ALFA)
        self.filtro2 = FiltroCausal(ALFA)

        self.leituras = 0
        self.falhas = 0

        # ------------------------------------------------------
        # Controle da coleta
        # ------------------------------------------------------

        self.estado = "pronto"
        # Estados possíveis:
        # pronto, contagem, gravando

        self.instante_estado = 0.0
        self.inicio_gravacao = 0.0

        self.buffer_amostra = []
        self.numero_amostra = 0

        # Contador global de linhas no CSV mestre.
        self.indice_global = 0

        # ------------------------------------------------------
        # Criação da sessão
        # ------------------------------------------------------

        data_sessao = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.id_sessao = f"S_{data_sessao}"

        self.pasta_sessao = PASTA_DADOS / self.id_sessao
        self.pasta_sessao.mkdir(parents=True, exist_ok=True)

        self.arquivo_dados = self.pasta_sessao / "dados.csv"
        self.arquivo_segmentos = self.pasta_sessao / "segmentos.csv"

        self.inicializar_csv()

        # ------------------------------------------------------
        # Interface gráfica
        # ------------------------------------------------------

        self.fig = plt.figure(figsize=(13, 8))

        # Reservamos espaço inferior para os campos de coleta.
        self.fig.subplots_adjust(bottom=0.28)

        grade = self.fig.add_gridspec(
            2, 2,
            width_ratios=[1, 1.2]
        )

        self.ax_xy = self.fig.add_subplot(grade[:, 0])
        self.ax_d1 = self.fig.add_subplot(grade[0, 1])
        self.ax_d2 = self.fig.add_subplot(grade[1, 1], sharex=self.ax_d1)

        self.configurar_grafico_xy()
        self.configurar_graficos_temporais()
        self.criar_controles()

        # Evento de teclado.
        self.fig.canvas.mpl_connect(
            "key_press_event",
            self.tecla
        )

        self.fig.tight_layout(rect=(0, 0.24, 1, 1))

    # ==========================================================
    # 5. CRIAÇÃO DOS CSVs
    # ==========================================================

    def inicializar_csv(self):

        # CSV principal: cada linha representa uma leitura.
        with open(
            self.arquivo_dados,
            "w",
            newline="",
            encoding="utf-8"
        ) as arquivo:

            escritor = csv.writer(arquivo)

            escritor.writerow([
                "tempo_ms",
                "t_rel_ms",
                "d1",
                "d2",
                "gesto",
                "pessoa",
                "objeto",
                "sessao",
                "amostra"
            ])

        # CSV secundário: índice de cada gesto coletado.
        with open(
            self.arquivo_segmentos,
            "w",
            newline="",
            encoding="utf-8"
        ) as arquivo:

            escritor = csv.writer(arquivo)

            escritor.writerow([
                "amostra",
                "gesto",
                "pessoa",
                "objeto",
                "sessao",
                "modo",
                "inicio_idx",
                "fim_idx",
                "inicio_ms",
                "fim_ms",
                "duracao_ms",
                "quantidade_leituras",
                "falhas",
                "percentual_falhas"
            ])

    # ==========================================================
    # 6. CONFIGURAÇÃO DOS GRÁFICOS
    # ==========================================================

    def configurar_grafico_xy(self):

        ax = self.ax_xy

        ax.set_xlim(*LIM_CM)
        ax.set_ylim(*LIM_CM)
        ax.set_aspect("equal")

        ax.grid(alpha=0.3)

        ax.set_xlabel("d1 (cm) → x")
        ax.set_ylabel("d2 (cm) → y")

        # Área de referência para realização do gesto.
        ax.add_patch(
            Rectangle(
                (AREA_UTIL[0], AREA_UTIL[0]),
                AREA_UTIL[1] - AREA_UTIL[0],
                AREA_UTIL[1] - AREA_UTIL[0],
                fill=False,
                linestyle="--",
                color="gray"
            )
        )

        # Trajetória bruta em cinza.
        self.bruto_xy, = ax.plot(
            [], [],
            ".",
            color="lightgray",
            markersize=5
        )

        # Trajetória filtrada.
        self.trilha, = ax.plot(
            [], [],
            "-",
            color="C0",
            alpha=0.4
        )

        # Pontos coloridos conforme o tempo.
        self.pontos = ax.scatter([], [], s=25)
        self.pontos.set_cmap("viridis")
        self.pontos.set_clim(0, JANELA_S)

        # Último ponto filtrado.
        self.atual, = ax.plot(
            [], [],
            "o",
            color="red",
            markersize=10
        )

    def configurar_graficos_temporais(self):

        for eixo, nome in (
            (self.ax_d1, "d1 (cm)"),
            (self.ax_d2, "d2 (cm)")
        ):
            eixo.set_ylim(*LIM_CM)
            eixo.set_ylabel(nome)
            eixo.grid(alpha=0.3)

        self.ax_d1.set_xlim(-JANELA_S, 0)
        self.ax_d2.set_xlabel(
            "tempo (s) — 0 = leitura mais recente"
        )

        # Leituras brutas.
        self.b1, = self.ax_d1.plot(
            [], [], ".",
            color="lightgray",
            markersize=5
        )

        self.b2, = self.ax_d2.plot(
            [], [], ".",
            color="lightgray",
            markersize=5
        )

        # Leituras filtradas.
        self.l1, = self.ax_d1.plot([], [], "-", color="C0")
        self.l2, = self.ax_d2.plot([], [], "-", color="C1")

        # Marca leituras inválidas.
        self.x1, = self.ax_d1.plot([], [], "rx")
        self.x2, = self.ax_d2.plot([], [], "rx")

    # ==========================================================
    # 7. CONTROLES DA COLETA
    # ==========================================================

    def criar_controles(self):

        # Campos para identificação dos dados.
        ax_gesto = self.fig.add_axes([0.08, 0.15, 0.25, 0.045])
        ax_pessoa = self.fig.add_axes([0.38, 0.15, 0.20, 0.045])
        ax_objeto = self.fig.add_axes([0.63, 0.15, 0.20, 0.045])

        self.txt_gesto = TextBox(ax_gesto, "Gesto: ", initial="circulo")
        self.txt_pessoa = TextBox(ax_pessoa, "Pessoa: ", initial="P01")
        self.txt_objeto = TextBox(ax_objeto, "Objeto: ", initial="O01")

        # Texto de status.
        self.status = self.fig.text(
            0.5, 0.08,
            "PRONTO | R = iniciar coleta",
            ha="center",
            fontsize=12,
            weight="bold"
        )

        self.fig.text(
            0.5, 0.035,
            "R: coletar | C: limpar | S: estatísticas | Q: sair",
            ha="center",
            fontsize=10
        )

        self.fig.text(
            0.5, 0.005,
            f"Sessão: {self.id_sessao}",
            ha="center",
            fontsize=8
        )

    # ==========================================================
    # 8. CONTROLE POR TECLADO
    # ==========================================================

    def tecla(self, evento):

        tecla = evento.key

        # Não executar atalhos se o usuário estiver digitando
        # dentro de um campo de texto.
        if evento.inaxes in (
            self.txt_gesto.ax,
            self.txt_pessoa.ax,
            self.txt_objeto.ax
        ):
            return

        if tecla == "r":
            self.iniciar_coleta()

        elif tecla == "c":
            self.limpar_graficos()

        elif tecla == "s":
            self.estatisticas()

        elif tecla == "q":
            plt.close(self.fig)

    # ==========================================================
    # 9. INÍCIO DA COLETA
    # ==========================================================

    def iniciar_coleta(self):

        # Não permite iniciar outra coleta enquanto uma estiver ativa.
        if self.estado != "pronto":
            print("Já existe uma coleta em andamento.")
            return

        # Verifica os campos de identificação.
        gesto = self.txt_gesto.text.strip()
        pessoa = self.txt_pessoa.text.strip()
        objeto = self.txt_objeto.text.strip()

        if not gesto or not pessoa or not objeto:
            self.status.set_text(
                "Preencha gesto, pessoa e objeto!"
            )
            return

        self.numero_amostra += 1

        self.buffer_amostra = []

        self.estado = "contagem"

        self.instante_estado = (
            time.monotonic() + CONTAGEM_REGRESSIVA
        )

        self.status.set_text(
            f"Prepare-se: {gesto.upper()} | "
            f"Amostra {self.numero_amostra}"
        )

        print(
            f"\nPreparando amostra {self.numero_amostra}: "
            f"{gesto}..."
        )

    # ==========================================================
    # 10. RECEPÇÃO E ARMAZENAMENTO DAS LEITURAS
    # ==========================================================

    def ler_serial(self):

        # Lê todas as linhas que já chegaram na serial.
        while self.ser.in_waiting:

            bruto = self.ser.readline()

            # Ignora linhas incompletas.
            if not bruto.endswith(b"\n"):
                continue

            linha = bruto.decode(
                "utf-8",
                errors="ignore"
            )

            amostra = interpretar(linha)

            if amostra is None:
                continue

            t, a_bruto, b_bruto, a, b = amostra

            # Histórico do monitoramento.
            self.t.append(t)
            self.r1.append(a)
            self.r2.append(b)

            self.f1.append(self.filtro1.passo(a))
            self.f2.append(self.filtro2.passo(b))

            self.leituras += 2

            self.falhas += (
                int(np.isnan(a))
                + int(np.isnan(b))
            )

            # Limita o tamanho do histórico gráfico.
            while len(self.t) > MAX_PONTOS_GRAFICO:

                self.t.popleft()
                self.r1.popleft()
                self.r2.popleft()
                self.f1.popleft()
                self.f2.popleft()

            # Se estiver gravando, adiciona ao buffer.
            if self.estado == "gravando":

                agora = time.monotonic()

                t_rel = int(
                    (agora - self.inicio_gravacao) * 1000
                )

                # Guarda valores brutos, incluindo -1.
                self.buffer_amostra.append([
                    t,
                    t_rel,
                    a_bruto,
                    b_bruto
                ])

    # ==========================================================
    # 11. SALVAMENTO DA AMOSTRA
    # ==========================================================

    def salvar_amostra(self):

        if not self.buffer_amostra:
            self.status.set_text(
                "Nenhuma leitura capturada. Amostra descartada."
            )
            self.estado = "pronto"
            return

        gesto = self.txt_gesto.text.strip()
        pessoa = self.txt_pessoa.text.strip()
        objeto = self.txt_objeto.text.strip()

        # Índice inicial no CSV mestre.
        inicio_idx = self.indice_global

        # O instante inicial da gravação (millis do ESP32).
        inicio_ms = self.buffer_amostra[0][0]

        # Instante final.
        fim_ms = self.buffer_amostra[-1][0]

        # Índice final exclusivo: a próxima linha a ser escrita.
        fim_idx = inicio_idx + len(self.buffer_amostra)

        # ------------------------------------------------------
        # Escreve os dados brutos no CSV mestre.
        # ------------------------------------------------------

        with open(
            self.arquivo_dados,
            "a",
            newline="",
            encoding="utf-8"
        ) as arquivo:

            escritor = csv.writer(arquivo)

            for t, t_rel, d1, d2 in self.buffer_amostra:

                escritor.writerow([
                    t,
                    t_rel,
                    d1,
                    d2,
                    gesto,
                    pessoa,
                    objeto,
                    self.id_sessao,
                    self.numero_amostra
                ])

        # ------------------------------------------------------
        # Calcula quantidade de falhas.
        # ------------------------------------------------------


        # Total de leituras de sensores:
        # cada amostra contém duas leituras (d1 e d2).
        total_leituras = len(self.buffer_amostra) * 2

        # Conta cada sensor inválido separadamente.
        falhas = sum(
            int(d1 < 0) + int(d2 < 0)
            for _, _, d1, d2 in self.buffer_amostra
        )

        percentual_falhas = (
            100 * falhas / total_leituras
            if total_leituras > 0
            else 0
        )

        duracao_ms = (
            self.buffer_amostra[-1][1]
            - self.buffer_amostra[0][1]
        )

        # ------------------------------------------------------
        # Registra o segmento no CSV de índices.
        # ------------------------------------------------------

        with open(
            self.arquivo_segmentos,
            "a",
            newline="",
            encoding="utf-8"
        ) as arquivo:

            escritor = csv.writer(arquivo)

            escritor.writerow([
                self.numero_amostra,
                gesto,
                pessoa,
                objeto,
                self.id_sessao,
                "normal",
                inicio_idx,
                fim_idx,
                inicio_ms,
                fim_ms,
                duracao_ms,
                len(self.buffer_amostra),
                falhas,
                round(percentual_falhas, 2)
            ])

        # Atualiza o índice global para a próxima amostra.
        self.indice_global = fim_idx

        print(
            f"Salvo: amostra {self.numero_amostra} | "
            f"{len(self.buffer_amostra)} leituras | "
            f"{percentual_falhas:.1f}% de falhas"
        )

        self.status.set_text(
            f"SALVO: {gesto} | Amostra {self.numero_amostra} | "
            f"{len(self.buffer_amostra)} leituras | "
            f"Falhas: {percentual_falhas:.1f}% | R = próxima"
        )

        self.estado = "pronto"
        self.buffer_amostra = []

    # ==========================================================
    # 12. LIMPAR HISTÓRICO VISUAL
    # ==========================================================

    def limpar_graficos(self):

        # Limpa somente os gráficos e os filtros.
        # Não apaga nenhum arquivo de coleta.
        for fila in (
            self.t,
            self.r1,
            self.r2,
            self.f1,
            self.f2
        ):
            fila.clear()

        self.filtro1.reiniciar()
        self.filtro2.reiniciar()

        self.leituras = 0
        self.falhas = 0

        print("Histórico visual limpo.")

    # ==========================================================
    # 13. ESTATÍSTICAS
    # ==========================================================

    def estatisticas(self):

        for nome, fila in (
            ("d1", self.r1),
            ("d2", self.r2)
        ):

            valores = np.array(fila)
            valores = valores[~np.isnan(valores)]

            if len(valores) > 3:

                print(
                    f"{nome} bruto: "
                    f"média = {valores.mean():.2f} cm | "
                    f"desvio = {valores.std():.2f} cm | "
                    f"n = {len(valores)}"
                )

    # ==========================================================
    # 14. ATUALIZAÇÃO DOS GRÁFICOS E ESTADOS DA COLETA
    # ==========================================================

    def atualizar(self, _quadro=None):

        # Recebe dados da serial continuamente.
        self.ler_serial()

        # ------------------------------------------------------
        # Máquina de estados da coleta
        # ------------------------------------------------------

        agora = time.monotonic()

        if self.estado == "contagem":

            restante = self.instante_estado - agora

            if restante > 0:

                self.status.set_text(
                    f"Prepare-se! Gravação em {int(restante) + 1}"
                )

            else:

                # A contagem terminou.
                # Começa a captura.
                self.estado = "gravando"
                self.inicio_gravacao = agora

                self.status.set_text(
                    f"GRAVANDO: {self.txt_gesto.text.upper()}!"
                )

                print("Gravando gesto...")

        elif self.estado == "gravando":

            duracao = agora - self.inicio_gravacao

            restante = DURACAO_COLETA - duracao

            if restante > 0:

                self.status.set_text(
                    f"GRAVANDO... {restante:.1f} s"
                )

            else:

                # Finaliza e salva automaticamente.
                self.salvar_amostra()

        # ------------------------------------------------------
        # Se ainda não há dados, não há gráficos para atualizar.
        # ------------------------------------------------------

        if not self.t:
            return ()

        t = np.array(self.t)
        ra = np.array(self.r1)
        rb = np.array(self.r2)
        fa = np.array(self.f1)
        fb = np.array(self.f2)

        # Tempo relativo ao ponto mais recente.
        rel = (t - t[-1]) / 1000.0

        # ------------------------------------------------------
        # Gráficos temporais
        # ------------------------------------------------------

        self.b1.set_data(rel, ra)
        self.b2.set_data(rel, rb)

        self.l1.set_data(rel, fa)
        self.l2.set_data(rel, fb)

        # Marca as falhas nos gráficos.
        falha_a = np.isnan(ra)
        falha_b = np.isnan(rb)

        self.x1.set_data(
            rel[falha_a],
            np.full(falha_a.sum(), LIM_CM[0] + 1)
        )

        self.x2.set_data(
            rel[falha_b],
            np.full(falha_b.sum(), LIM_CM[0] + 1)
        )

        # ------------------------------------------------------
        # Gráfico XY
        # ------------------------------------------------------

        ok_bruto = ~falha_a & ~falha_b

        self.bruto_xy.set_data(
            ra[ok_bruto],
            rb[ok_bruto]
        )

        ok_filtrado = ~np.isnan(fa) & ~np.isnan(fb)

        self.pontos.set_offsets(
            np.c_[fa[ok_filtrado], fb[ok_filtrado]]
        )

        self.pontos.set_array(-rel[ok_filtrado])

        self.trilha.set_data(
            fa[ok_filtrado],
            fb[ok_filtrado]
        )

        if ok_filtrado.any():

            self.atual.set_data(
                [fa[ok_filtrado][-1]],
                [fb[ok_filtrado][-1]]
            )

        else:

            self.atual.set_data([], [])

        # ------------------------------------------------------
        # Porcentagem geral de falhas no monitoramento
        # ------------------------------------------------------

        pct = (
            100 * self.falhas / max(self.leituras, 1)
        )

        self.ax_xy.set_title(
            f"Trajetória d1 × d2 | "
            f"Leituras inválidas: {pct:.1f}%"
        )

        return ()


# ==============================================================
# 15. INICIALIZAÇÃO
# ==============================================================

def main():

    porta = sys.argv[1] if len(sys.argv) > 1 else PORTA

    print("=== MONITORAMENTO E COLETA DE GESTOS ===")
    print(f"Porta serial: {porta}")
    print("Pressione R para iniciar uma coleta.")
    print("Pressione Q para fechar.")

    try:

        with serial.Serial(
            porta,
            BAUDRATE,
            timeout=0.05
        ) as ser:

            # Aguarda o ESP32 estabilizar após abrir a serial.
            time.sleep(2)
            ser.reset_input_buffer()

            painel = Painel(ser)

            # Atualiza a janela periodicamente.
            anim = FuncAnimation(
                painel.fig,
                painel.atualizar,
                interval=50,
                cache_frame_data=False
            )

            plt.show()

    except serial.SerialException as erro:

        print(f"Erro na comunicação serial: {erro}")

    except KeyboardInterrupt:

        print("\nPrograma interrompido.")


if __name__ == "__main__":
    main()
