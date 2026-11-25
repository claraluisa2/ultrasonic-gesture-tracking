
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

PASTA_DADOS = Path(__file__).resolve().parent.parent / "dados"

# Para conferir os nomes que existem de verdade:
print([f.name for f in PASTA_DADOS.glob("*.csv")])

df = pd.read_csv(PASTA_DADOS / "O_1_1_20261125_151552.csv")

plt.figure(figsize=(8, 8))

plt.plot(df["d1"], df["d2"], alpha=0.5)
plt.scatter(
    df["d1"],
    df["d2"],
    c=df["t_rel_ms"],
    cmap="viridis"
)

plt.xlabel("d1")
plt.ylabel("d2")
plt.title("Trajetória do gesto circular")
plt.colorbar(label="Tempo (ms)")
plt.axis("equal")
plt.grid()
plt.show()
