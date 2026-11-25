"""Visualização em tempo real dos dois HC-SR04 (serial: tempo_ms,d1,d2).

Uso:    python ao_vivo.py COM4
Teclas: 'c' limpa a janela | 'q' fecha

Esquerda : trajetória (d1 -> x, d2 -> y). Ponto vermelho = leitura mais recente;
           a cor dos pontos indica a idade (roxo = recente, amarelo = antigo).
Direita  : d1 e d2 no tempo. Os 'x' vermelhos embaixo são leituras inválidas (-1).
"""
import collections
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

PORTA = "COM4"
BAUDRATE = 115200
JANELA_S = 4.0        # quantos segundos de histórico mostrar
LIM_CM = (0, 40)      # alcance útil dos sensores
AREA_UTIL = (10, 30)  # caixa tracejada onde o gesto deve acontecer (cm)


def interpretar(linha):
    """'tempo_ms,d1,d2' -> (t, d1, d2) com NaN no lugar de -1; None se a linha for inválida."""
    partes = linha.strip().split(",")
    if len(partes) != 3:
        return None
    try:
        t, a, b = int(partes[0]), float(partes[1]), float(partes[2])
    except ValueError:
        return None
    return t, (a if a > 0 else np.nan), (b if b > 0 else np.nan)


class Painel:
    def __init__(self, ser):
        self.ser = ser
        self.t = collections.deque()
        self.d1 = collections.deque()
        self.d2 = collections.deque()
        self.leituras = 0
        self.falhas = 0

        self.fig = plt.figure(figsize=(12, 6))
        grade = self.fig.add_gridspec(2, 2, width_ratios=[1, 1.2])
        self.ax_xy = self.fig.add_subplot(grade[:, 0])
        self.ax_d1 = self.fig.add_subplot(grade[0, 1])
        self.ax_d2 = self.fig.add_subplot(grade[1, 1], sharex=self.ax_d1)

        # trajetória
        ax = self.ax_xy
        ax.set_xlim(*LIM_CM)
        ax.set_ylim(*LIM_CM)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("d1 (cm)  →  x")
        ax.set_ylabel("d2 (cm)  →  y")
        ax.add_patch(
            Rectangle(
                (AREA_UTIL[0], AREA_UTIL[0]),
                AREA_UTIL[1] - AREA_UTIL[0],
                AREA_UTIL[1] - AREA_UTIL[0],
                fill=False, linestyle="--", color="gray",
            )
        )
        (self.trilha,) = ax.plot([], [], "-", color="C0", alpha=0.3)
        self.pontos = ax.scatter([], [], s=25)
        self.pontos.set_cmap("viridis")
        self.pontos.set_clim(0, JANELA_S)
        (self.atual,) = ax.plot([], [], "o", color="red", ms=10)

        # sinais no tempo
        for eixo, nome in ((self.ax_d1, "d1 (cm)"), (self.ax_d2, "d2 (cm)")):
            eixo.set_ylim(*LIM_CM)
            eixo.set_ylabel(nome)
            eixo.grid(alpha=0.3)
        self.ax_d1.set_xlim(-JANELA_S, 0)
        self.ax_d2.set_xlabel("tempo (s) — 0 = leitura mais recente")
        (self.l1,) = self.ax_d1.plot([], [], ".-", color="C0")
        (self.l2,) = self.ax_d2.plot([], [], ".-", color="C1")
        (self.f1,) = self.ax_d1.plot([], [], "rx")
        (self.f2,) = self.ax_d2.plot([], [], "rx")

        self.fig.canvas.mpl_connect("key_press_event", self.tecla)
        self.fig.tight_layout()

    def tecla(self, evento):
        if evento.key == "c":
            self.t.clear(); self.d1.clear(); self.d2.clear()
            self.leituras = self.falhas = 0
        elif evento.key == "q":
            plt.close(self.fig)

    def ler_serial(self):
        while self.ser.in_waiting:
            bruto = self.ser.readline()
            if not bruto.endswith(b"\n"):   # linha cortada: ignora
                continue
            amostra = interpretar(bruto.decode("utf-8", errors="ignore"))
            if amostra is None:
                continue
            t, a, b = amostra
            self.t.append(t); self.d1.append(a); self.d2.append(b)
            self.leituras += 2
            self.falhas += int(np.isnan(a)) + int(np.isnan(b))

        while self.t and self.t[-1] - self.t[0] > JANELA_S * 1000:
            self.t.popleft(); self.d1.popleft(); self.d2.popleft()

    def atualizar(self, _quadro=None):
        self.ler_serial()
        if not self.t:
            return ()

        t = np.array(self.t)
        a = np.array(self.d1)
        b = np.array(self.d2)
        rel = (t - t[-1]) / 1000.0

        self.l1.set_data(rel, a)
        self.l2.set_data(rel, b)
        falha_a, falha_b = np.isnan(a), np.isnan(b)
        self.f1.set_data(rel[falha_a], np.full(falha_a.sum(), LIM_CM[0] + 1))
        self.f2.set_data(rel[falha_b], np.full(falha_b.sum(), LIM_CM[0] + 1))

        ok = ~falha_a & ~falha_b
        self.pontos.set_offsets(np.c_[a[ok], b[ok]])
        self.pontos.set_array(-rel[ok])
        self.trilha.set_data(a[ok], b[ok])
        if ok.any():
            self.atual.set_data([a[ok][-1]], [b[ok][-1]])
        else:
            self.atual.set_data([], [])

        pct = 100 * self.falhas / max(self.leituras, 1)
        self.ax_xy.set_title(f"trajetória (d1 × d2)  |  leituras inválidas: {pct:.0f}%")
        return ()


def main():
    porta = sys.argv[1] if len(sys.argv) > 1 else PORTA
    with serial.Serial(porta, BAUDRATE, timeout=0.05) as ser:
        time.sleep(2)
        ser.reset_input_buffer()
        painel = Painel(ser)
        anim = FuncAnimation(  # noqa: F841 (precisa continuar referenciada)
            painel.fig, painel.atualizar, interval=50, cache_frame_data=False
        )
        plt.show()


if __name__ == "__main__":
    main()