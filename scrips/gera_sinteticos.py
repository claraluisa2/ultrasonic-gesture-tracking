"""
Gera o dataset de treino a partir das sessões gravadas pelo ao_vivo.py.

    python gerar_sinteticos.py                 # usa ../dados e gera 50 sintéticas por amostra real
    python gerar_sinteticos.py --n 100 --pasta C:/caminho/dados

Fluxo:
  1. Lê todo dados.csv encontrado dentro da pasta (uma ou várias sessões).
  2. Cada amostra real passa por: limpar (-1 -> interpolação) -> recortar (só o trecho com
     movimento) -> extrair features (16 pontos x,y no tempo + duração = 33 números).
  3. Separa amostras REAIS em treino e validação (antes de gerar qualquer sintética).
  4. Cada amostra real de TREINO gera N variações sintéticas: o caminho é deformado
     (rotação, escala, deslocamento, velocidade), reamostrado na grade de 70 ms do sensor,
     recebe ruído, picos e falhas, e passa pelo MESMO pipeline das amostras reais.
  5. Salva dataset.npz, dataset.csv e diagnostico.png em <pasta>/dataset/.

A validação usa SÓ amostras reais. Sintéticas nunca entram nela.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# ============================================================
# CONFIGURAÇÕES
# ============================================================
PASTA_DADOS = Path(__file__).resolve().parent.parent / "dados"

PERIODO_MS = 70           # período de amostragem do firmware
N_PONTOS = 16             # pontos por gesto na entrada da rede
ESCALA_CM = 50.0          # x,y são divididos por isto (fica em ~0..1)
JANELA_S = 3.0            # duração da janela de gravação (normaliza a duração)
CLASSE_NENHUMA = "nenhuma"

# --- recorte do gesto (mesma ideia do detector de movimento do firmware) ---
VEL_MIN_CM = 1.5          # cm por amostra, após mediana de 3, para contar como movimento
MARGEM = 1                # amostras mantidas antes/depois do movimento
MIN_AMOSTRAS = 8          # gestos mais curtos que isto (~0,56 s) são descartados
FRACAO_VALIDA_MIN = 0.70  # amostras com mais falhas que isso são descartadas

# --- deformações do caminho (augmentation) ---
ROT_MAX_GRAUS = 10
ESCALA_FAIXA = (0.85, 1.20)       # x e y sorteados separadamente
DESLOC_MAX_CM = 4.0
DURACAO_FAIXA = (0.75, 1.35)      # <1 = gesto mais rápido, >1 = mais lento
WARP_AMP = 0.12                   # variação suave da velocidade ao longo do gesto
PAD_AMOSTRAS = (2, 10)            # objeto parado antes/depois do gesto

# --- modelo do sensor (sorteado a cada amostra sintética) ---
RUIDO_CM = (0.2, 0.8)             # desvio-padrão do ruído gaussiano
P_PICO_MAX = 0.06                 # probabilidade de leitura com pico isolado
PICO_CM = (4.0, 12.0)
P_FALHA_MAX = 0.06                # real medido: 0 a ~6% por amostra
DIST_MIN_CM, DIST_MAX_CM = 2.0, 45.0   # fora disso o sensor devolve -1


# ============================================================
# PIPELINE (idêntico para amostras reais e sintéticas)
# ============================================================
def mediana3(v):
    p = np.pad(v, 1, mode="edge")
    return np.median(sliding_window_view(p, 3), axis=1)


def limpar(t, d1, d2):
    """-1 vira buraco e o buraco é interpolado. None se faltar leitura demais."""
    saida = []
    for d in (d1, d2):
        d = np.where(d > 0, d, np.nan)
        ok = ~np.isnan(d)
        if ok.mean() < FRACAO_VALIDA_MIN:
            return None
        saida.append(np.interp(t, t[ok], d[ok]))
    return saida


def recortar(x, y):
    """Índices (i0, i1) do trecho com movimento, ou None se não houver."""
    vel = np.hypot(np.diff(mediana3(x)), np.diff(mediana3(y)))
    ativo = np.flatnonzero(vel > VEL_MIN_CM)
    if ativo.size < 3:           # 1-2 saltos isolados são ruído, não gesto
        return None
    i0 = max(ativo[0] - MARGEM, 0)
    i1 = min(ativo[-1] + 1 + MARGEM, len(x) - 1)
    return i0, i1 + 1


def preparar(t, d1, d2, classe):
    """Dados crus -> caminho limpo e recortado {t, x, y}. Retorna (dict, motivo_descarte)."""
    limpo = limpar(t, d1, d2)
    if limpo is None:
        return None, "muitas leituras inválidas"
    x, y = limpo
    corte = recortar(x, y)
    if corte is None:
        if classe != CLASSE_NENHUMA:
            return None, "sem movimento"
        corte = (0, len(x))               # 'nenhuma' pode ser mesmo um objeto parado
    i0, i1 = corte
    if i1 - i0 < MIN_AMOSTRAS:
        return None, "gesto curto demais"
    return {"t": t[i0:i1] - t[i0], "x": x[i0:i1], "y": y[i0:i1]}, None


def extrair_features(p):
    """16 pontos (x,y) igualmente espaçados no tempo + duração -> 33 números."""
    tn = np.linspace(p["t"][0], p["t"][-1], N_PONTOS)
    pts = np.column_stack([np.interp(tn, p["t"], p["x"]),
                           np.interp(tn, p["t"], p["y"])]) / ESCALA_CM
    dur = p["t"][-1] / 1000.0 / JANELA_S
    return np.concatenate([pts.ravel(), [dur]]).astype(np.float32)


# ============================================================
# AMOSTRAS SINTÉTICAS
# ============================================================
def sintetizar(p, rng):
    """Caminho real limpo -> leituras cruas simuladas (t_ms, d1, d2; -1 = falha)."""
    t, x, y = p["t"], mediana3(p["x"]), mediana3(p["y"])   # tira picos reais antes de recriar ruído

    # geometria
    cx, cy = x.mean(), y.mean()
    ang = np.deg2rad(rng.uniform(-ROT_MAX_GRAUS, ROT_MAX_GRAUS))
    u = (x - cx) * rng.uniform(*ESCALA_FAIXA)
    v = (y - cy) * rng.uniform(*ESCALA_FAIXA)
    xn = cx + rng.uniform(-DESLOC_MAX_CM, DESLOC_MAX_CM) + u * np.cos(ang) - v * np.sin(ang)
    yn = cy + rng.uniform(-DESLOC_MAX_CM, DESLOC_MAX_CM) + u * np.sin(ang) + v * np.cos(ang)

    # tempo: duração nova + variação suave de velocidade (monótona, pois |amp| < 1/pi)
    s = t / t[-1]
    duracao = t[-1] * rng.uniform(*DURACAO_FAIXA)
    fase = rng.uniform(0, PERIODO_MS)
    tn = np.arange(fase, duracao, PERIODO_MS)
    tau = tn / duracao
    amp = rng.uniform(-WARP_AMP, WARP_AMP)
    g = np.clip(tau + amp * np.sin(np.pi * tau), 0, 1)
    xt, yt = np.interp(g, s, xn), np.interp(g, s, yn)

    # objeto parado antes e depois (como nas gravações de 3 s)
    a, b = rng.integers(PAD_AMOSTRAS[0], PAD_AMOSTRAS[1] + 1, size=2)
    xt = np.concatenate([np.full(a, xt[0]), xt, np.full(b, xt[-1])])
    yt = np.concatenate([np.full(a, yt[0]), yt, np.full(b, yt[-1])])
    t_ms = np.arange(len(xt)) * PERIODO_MS

    # modelo do sensor
    sigma = rng.uniform(*RUIDO_CM)
    p_pico = rng.uniform(0, P_PICO_MAX)
    p_falha = rng.uniform(0, P_FALHA_MAX)
    saida = []
    for d in (xt, yt):
        d = d + rng.normal(0, sigma, d.size)
        pico = rng.random(d.size) < p_pico
        d[pico] += rng.choice([-1, 1], pico.sum()) * rng.uniform(*PICO_CM, pico.sum())
        d = np.round(d, 1)
        d[(d < DIST_MIN_CM) | (d > DIST_MAX_CM)] = -1     # fora do alcance
        d[rng.random(d.size) < p_falha] = -1              # falha de eco
        saida.append(d)
    return t_ms.astype(float), saida[0], saida[1]


# ============================================================
# LEITURA, DIVISÃO E SAÍDAS
# ============================================================
def carregar_amostras(pasta):
    arquivos = sorted(pasta.rglob("dados.csv"))
    if not arquivos:
        raise SystemExit(f"Nenhum dados.csv encontrado em {pasta}")
    amostras = []
    for arq in arquivos:
        df = pd.read_csv(arq)
        for (sessao, num), g in df.groupby(["sessao", "amostra"], sort=True):
            amostras.append({
                "id": f"{sessao}:{num}",
                "sessao": sessao,
                "classe": str(g["gesto"].iloc[0]),
                # tempo do ESP32 (millis). O t_rel_ms do CSV é do PC e sai repetido.
                "t": (g["tempo_ms"] - g["tempo_ms"].iloc[0]).to_numpy(float),
                "d1": g["d1"].to_numpy(float),
                "d2": g["d2"].to_numpy(float),
            })
    return amostras


def dividir(validas, frac_val, rng):
    """Sorteia amostras REAIS para validação, por classe. Sempre sobra ao menos 1 no treino."""
    treino, val = [], []
    for classe in sorted({a["classe"] for a in validas}):
        grupo = [a for a in validas if a["classe"] == classe]
        rng.shuffle(grupo)
        n_val = int(round(frac_val * len(grupo))) if len(grupo) >= 4 else 0
        val += grupo[:n_val]
        treino += grupo[n_val:]
    return treino, val


def desempacotar(f):
    pts = f[:-1].reshape(N_PONTOS, 2) * ESCALA_CM
    return pts[:, 0], pts[:, 1]


def diagnostico(linhas, classes, caminho):
    fig, axs = plt.subplots(len(classes), 3, figsize=(15, 4.5 * len(classes)), squeeze=False)
    for i, classe in enumerate(classes):
        ax_xy, ax_x, ax_y = axs[i]
        for tipo, cor, alfa, lw in (("sintetica", "gray", 0.12, 1), ("real", "C3", 0.9, 1.8)):
            for l in linhas:
                if l["classe"] != classe or l["tipo"] != tipo:
                    continue
                x, y = desempacotar(l["f"])
                ax_xy.plot(x, y, color=cor, alpha=alfa, lw=lw)
                k = np.linspace(0, 1, N_PONTOS)
                ax_x.plot(k, x, color=cor, alpha=alfa, lw=lw)
                ax_y.plot(k, y, color=cor, alpha=alfa, lw=lw)
        ax_xy.set_title(f"{classe}: trajetória d1 × d2 (vermelho = real, cinza = sintética)")
        ax_x.set_title("d1 ao longo do gesto (tempo normalizado)")
        ax_y.set_title("d2 ao longo do gesto (tempo normalizado)")
        ax_xy.set_aspect("equal")
        for ax in axs[i]:
            ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho, dpi=70)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", type=Path, default=PASTA_DADOS)
    ap.add_argument("--n", type=int, default=50, help="sintéticas por amostra real de treino")
    ap.add_argument("--val", type=float, default=0.25, help="fração das amostras reais para validação")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    saida = args.pasta / "dataset"
    saida.mkdir(exist_ok=True)

    # 1. amostras reais -> caminho limpo
    validas, descartadas = [], []
    for a in carregar_amostras(args.pasta):
        p, motivo = preparar(a["t"], a["d1"], a["d2"], a["classe"])
        if p is None:
            descartadas.append((a["id"], a["classe"], motivo))
        else:
            a["p"] = p
            validas.append(a)

    print(f"{len(validas)} amostras reais válidas, {len(descartadas)} descartadas")
    for id_, classe, motivo in descartadas:
        print(f"  descartada {id_} ({classe}): {motivo}")
    if not validas:
        raise SystemExit("Nada para processar.")
    for a in validas:
        print(f"  {a['id']:>26} {a['classe']:<10} recorte: {len(a['p']['t'])} leituras, "
              f"{a['p']['t'][-1] / 1000:.2f} s")

    classes = sorted({a["classe"] for a in validas})
    if len(classes) < 2:
        print(f"\nATENÇÃO: só existe a classe {classes}. Uma rede precisa de pelo menos 2 classes "
              f"(incluindo '{CLASSE_NENHUMA}') para aprender a distinguir.")
    sessoes = {a["sessao"] for a in validas}
    if len(sessoes) < 2:
        print("ATENÇÃO: só há 1 sessão. A validação vem da mesma sessão e tende a ser otimista.")

    # 2. divisão ANTES de gerar sintéticas
    treino, val = dividir(validas, args.val, rng)

    # 3. features
    linhas = []
    for a in treino:
        linhas.append({"particao": "treino", "tipo": "real", "classe": a["classe"],
                       "origem": a["id"], "f": extrair_features(a["p"])})
        gerou = 0
        while gerou < args.n:
            t, d1, d2 = sintetizar(a["p"], rng)
            p, _ = preparar(t, d1, d2, a["classe"])
            if p is None:
                continue
            linhas.append({"particao": "treino", "tipo": "sintetica", "classe": a["classe"],
                           "origem": a["id"], "f": extrair_features(p)})
            gerou += 1
    for a in val:
        linhas.append({"particao": "validacao", "tipo": "real", "classe": a["classe"],
                       "origem": a["id"], "f": extrair_features(a["p"])})

    # 4. saídas
    indice = {c: i for i, c in enumerate(classes)}

    def pega(particao):
        sel = [l for l in linhas if l["particao"] == particao]
        return (np.array([l["f"] for l in sel], dtype=np.float32),
                np.array([indice[l["classe"]] for l in sel], dtype=np.int64),
                np.array([l["origem"] for l in sel]),
                np.array([l["tipo"] == "sintetica" for l in sel]))

    Xt, yt, ot, st = pega("treino")
    Xv, yv, ov, _ = pega("validacao")
    np.savez(saida / "dataset.npz", X_treino=Xt, y_treino=yt, origem_treino=ot, sintetica_treino=st,
             X_val=Xv, y_val=yv, origem_val=ov, classes=np.array(classes))

    colunas = [f"{eixo}{k}" for k in range(N_PONTOS) for eixo in ("x", "y")] + ["duracao"]
    tabela = pd.DataFrame([l["f"] for l in linhas], columns=colunas)
    for nome in ("origem", "classe", "tipo", "particao"):
        tabela.insert(0, nome, [l[nome] for l in linhas])
    tabela.to_csv(saida / "dataset.csv", index=False)

    diagnostico([l for l in linhas if l["particao"] == "treino"], classes, saida / "diagnostico.png")

    print(f"\nclasses: {classes}")
    print(f"treino: {len(Xt)} linhas ({int(st.sum())} sintéticas) | validação: {len(Xv)} linhas (só reais)")
    print(f"features por exemplo: {Xt.shape[1]}")
    print(f"arquivos em: {saida}")


if __name__ == "__main__":
    main()