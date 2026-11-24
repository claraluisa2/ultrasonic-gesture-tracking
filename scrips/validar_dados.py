import pandas as pd
from pathlib import Path

PASTA_DADOS = Path(__file__).resolve().parent.parent / "dados"

def avaliar(caminho):
    df = pd.read_csv(caminho)
    d = df[["d1", "d2"]]
    valido = d.where(d >= 0)                      # -1 vira NaN
    limpo = valido.rolling(5, center=True, min_periods=1).median()

    invalidas = (d < 0).mean().mean() * 100       # % de -1
    saltos = int((valido.diff().abs() > 8).sum().sum())
    repouso_ini = limpo.iloc[:5].std().max()
    repouso_fim = limpo.iloc[-5:].std().max()
    amplitude = (limpo.max() - limpo.min()).max()

    problemas = []
    if invalidas > 10:    problemas.append(f"{invalidas:.0f}% de -1")
    if saltos > 4:        problemas.append(f"{saltos} saltos bruscos")
    if repouso_ini > 0.5: problemas.append("sem repouso no início")
    if repouso_fim > 0.5: problemas.append("gesto cortado no final")
    if amplitude < 2:     problemas.append("quase sem movimento")

    status = "REVISAR" if problemas else "ok"
    print(f"{caminho.name}: {status} {', '.join(problemas)}")

for f in sorted(PASTA_DADOS.glob("*.csv")):
    avaliar(f)