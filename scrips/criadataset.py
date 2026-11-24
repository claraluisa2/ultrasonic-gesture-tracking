"""
Junta os CSVs de coleta em UM arquivo no formato longo: as leituras de cada
amostra ficam uma abaixo da outra, e as amostras ficam empilhadas em sequencia.
Os dados sao limpos e reamostrados numa grade uniforme de tempo.

Coloque este arquivo na mesma pasta do script de coleta (scrips/).
Uso:    python preparar_dataset.py
Saida:  ../dataset/dataset_longo.csv

IMPORTANTE: o firmware/inferencia precisa aplicar EXATAMENTE o mesmo
pre-processamento (DIST_MAX, MEDIANA, tratamento do -1, SUBTRAIR_BASE e a
grade de tempo de PASSO_MS), senao o modelo ve dados diferentes dos do treino.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------- Configuracao ----------------
RAIZ = Path(__file__).resolve().parent.parent
PASTA_DADOS = RAIZ / "dados"
ARQUIVO_SAIDA = RAIZ / "dataset" / "dataset_longo.csv"

PASSO_MS = 65          # intervalo entre leituras (loop do firmware)
JANELA_MS = 3000       # duracao de cada amostra
DIST_MAX = 40.0        # cm: acima disso (ou sem leitura longa) vale DIST_MAX
MEDIANA = 5            # tamanho do filtro de mediana (remove picos isolados)
MAX_INVALIDAS = 30.0   # % maxima de leituras -1 aceita por arquivo
SUBTRAIR_BASE = False  # True: subtrai a mediana do repouso inicial
N_REPOUSO = 8          # leituras iniciais usadas como repouso (~0,5 s)

N_PONTOS = JANELA_MS // PASSO_MS + 1
GRADE_MS = np.arange(N_PONTOS) * PASSO_MS
COLUNAS_NECESSARIAS = {"tempo_ms", "d1", "d2", "gesto"}


def limpar_sensor(serie):
    """-1 vira NaN, falhas curtas sao interpoladas, corta em DIST_MAX e
    remove picos isolados com filtro de mediana."""
    s = serie.astype(float).where(serie >= 0)
    s = s.interpolate(limit=3, limit_direction="both")
    s = s.fillna(DIST_MAX).clip(upper=DIST_MAX)
    return s.rolling(MEDIANA, center=True, min_periods=1).median()


def processar(caminho):
    """Retorna um DataFrame (uma linha por leitura) ou None se o arquivo for ignorado."""
    df = pd.read_csv(caminho)

    if not COLUNAS_NECESSARIAS.issubset(df.columns):
        print(f"[ignorado] {caminho.name}: faltam colunas (formato antigo?)")
        return None
    if len(df) < 10:
        print(f"[ignorado] {caminho.name}: poucas leituras ({len(df)})")
        return None

    t = (df["tempo_ms"] - df["tempo_ms"].iloc[0]).to_numpy()
    if t[-1] < JANELA_MS * 0.9:
        print(f"[ignorado] {caminho.name}: gravacao de {t[-1] / 1000:.1f} s, "
              f"menor que a janela de {JANELA_MS / 1000:.0f} s")
        return None

    pct_inv = float((df[["d1", "d2"]] < 0).mean().mean() * 100)
    if pct_inv > MAX_INVALIDAS:
        print(f"[ignorado] {caminho.name}: {pct_inv:.0f}% de leituras invalidas")
        return None

    sinais = []
    for col in ("d1", "d2"):
        s = limpar_sensor(df[col]).to_numpy()
        if SUBTRAIR_BASE:
            s = s - np.median(s[:N_REPOUSO])
        # Reamostra numa grade uniforme de PASSO_MS (compensa o jitter de 65/66 ms)
        sinais.append(np.interp(GRADE_MS, t, s))

    return pd.DataFrame({
        "amostra": caminho.stem,
        "gesto": df["gesto"].iloc[0],
        "pessoa": df["pessoa"].iloc[0] if "pessoa" in df else "",
        "objeto": df["objeto"].iloc[0] if "objeto" in df else "",
        "t_rel_ms": GRADE_MS,
        "d1": sinais[0].round(2),
        "d2": sinais[1].round(2),
    })


def main():
    arquivos = sorted(PASTA_DADOS.glob("*.csv"))
    partes = [r for r in (processar(f) for f in arquivos) if r is not None]

    if not partes:
        print("Nenhum arquivo valido encontrado em", PASTA_DADOS)
        return

    tabela = pd.concat(partes, ignore_index=True)
    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(ARQUIVO_SAIDA, index=False)

    print(f"\n{len(partes)} de {len(arquivos)} arquivos salvos em {ARQUIVO_SAIDA}")
    print(f"{len(tabela)} linhas ({N_PONTOS} leituras por amostra, passo de {PASSO_MS} ms)")
    print("\nAmostras por gesto e pessoa:")
    print(tabela.groupby(["gesto", "pessoa"])["amostra"].nunique().to_string())


if __name__ == "__main__":
    main()