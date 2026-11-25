"""Visualização em tempo real dos dois HC-SR04 (serial: tempo_ms,d1,d2), com filtro.

Uso:    python ao_vivo.py COM4
Teclas: 'c' limpa a janela
        's' imprime média e desvio-padrão do bruto (use com o objeto PARADO)
        'q' fecha

Cinza    = leitura bruta
Colorido = leitura filtrada (mediana de 3 + média exponencial)
Ponto vermelho = leitura filtrada mais recente. Os 'x' vermelhos embaixo = leituras inválidas (-1).
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
LIM_CM = (0, 100)     # alcance mostrado nos eixos (ajuste ao seu layout)
AREA_UTIL = (40, 60)  # caixa tracejada onde o gesto deve acontecer (cm)
ALFA = 0.5            # suavização: 1.0 = sem suavizar; menor = mais suave (e mais atrasado)


class FiltroCausal:
    """Mediana das últimas 3 leituras válidas (tira picos isolados) seguida de
    média exponencial (suaviza o tremor). Só usa o passado, então funciona igual
    em tempo real no ESP32. Leitura inválida continua inválida: não inventa dado."""

    def __init__(self, alfa):
        self.alfa = alfa
        self.reiniciar()

    def reiniciar(self):
        self.recentes = collections.deque(maxlen=3)
        self.y = np.nan
        self.falhas_seguidas = 0

    def passo(self, x):
        if np.isnan(x):
            self.falhas_seguidas += 1
            if self.falhas_seguidas >= 5:   # buraco longo: esquece o estado antigo
                self.reiniciar()
                self.falhas_seguidas = 5
            return np.nan
        self.falhas_seguidas = 0
        self.recentes.append(x)
        m = float(np.median(self.recentes))
        self.y = m if np.isnan(self.y) else self.alfa * m + (1 - self.alfa) * self.y
        return self.y


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
        self.r1, self.r2 = collections.deque(), collections.deque()  # bruto
        self.f1, self.f2 = collections.deque(), collections.deque()  # filtrado
        self.filtro1 = FiltroCausal(ALFA)
        self.filtro2 = FiltroCausal(ALFA)
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
        (self.bruto_xy,) = ax.plot([], [], ".", color="lightgray", ms=6)
        (self.trilha,) = ax.plot([], [], "-", color="C0", alpha=0.4)
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
        (self.b1,) = self.ax_d1.plot([], [], ".", color="lightgray", ms=5)
        (self.b2,) = self.ax_d2.plot([], [], ".", color="lightgray", ms=5)
        (self.l1,) = self.ax_d1.plot([], [], "-", color="C0")
        (self.l2,) = self.ax_d2.plot([], [], "-", color="C1")
        (self.x1,) = self.ax_d1.plot([], [], "rx")
        (self.x2,) = self.ax_d2.plot([], [], "rx")

        self.fig.canvas.mpl_connect("key_press_event", self.tecla)
        self.fig.tight_layout()

    def tecla(self, evento):
        if evento.key == "c":
            for fila in (self.t, self.r1, self.r2, self.f1, self.f2):
                fila.clear()
            self.filtro1.reiniciar()
            self.filtro2.reiniciar()
            self.leituras = self.falhas = 0
        elif evento.key == "s":
            for nome, fila in (("d1", self.r1), ("d2", self.r2)):
                v = np.array(fila)
                v = v[~np.isnan(v)]
                if len(v) > 3:
                    print(f"{nome} bruto: média {v.mean():.2f} cm | "
                          f"desvio-padrão {v.std():.2f} cm | n={len(v)}")
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
            self.t.append(t)
            self.r1.append(a); self.r2.append(b)
            self.f1.append(self.filtro1.passo(a))
            self.f2.append(self.filtro2.passo(b))
            self.leituras += 2
            self.falhas += int(np.isnan(a)) + int(np.isnan(b))

        while self.t and self.t[-1] - self.t[0] > JANELA_S * 1000:
            for fila in (self.t, self.r1, self.r2, self.f1, self.f2):
                fila.popleft()

    def atualizar(self, _quadro=None):
        self.ler_serial()
        if not self.t:
            return ()

        t = np.array(self.t)
        ra, rb = np.array(self.r1), np.array(self.r2)
        fa, fb = np.array(self.f1), np.array(self.f2)
        rel = (t - t[-1]) / 1000.0

        # sinais no tempo
        self.b1.set_data(rel, ra)
        self.b2.set_data(rel, rb)
        self.l1.set_data(rel, fa)
        self.l2.set_data(rel, fb)
        falha_a, falha_b = np.isnan(ra), np.isnan(rb)
        self.x1.set_data(rel[falha_a], np.full(falha_a.sum(), LIM_CM[0] + 1))
        self.x2.set_data(rel[falha_b], np.full(falha_b.sum(), LIM_CM[0] + 1))

        # trajetória: bruto (cinza) e filtrado (colorido)
        ok_r = ~falha_a & ~falha_b
        self.bruto_xy.set_data(ra[ok_r], rb[ok_r])
        ok = ~np.isnan(fa) & ~np.isnan(fb)
        self.pontos.set_offsets(np.c_[fa[ok], fb[ok]])
        self.pontos.set_array(-rel[ok])
        self.trilha.set_data(fa[ok], fb[ok])
        if ok.any():
            self.atual.set_data([fa[ok][-1]], [fb[ok][-1]])
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