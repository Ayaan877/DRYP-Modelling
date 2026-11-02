import os
import pandas as pd
import matplotlib.pyplot as plt

BASE = r"C:\Users\Ayaan Dutt\Desktop\WELL Labs\DRYP Modelling\AET\Watershed Bhuvan AET 5500m"
CSV = os.path.join(BASE, "aet_2025_5500m_dates.csv")

df = pd.read_csv(CSV, parse_dates=["date"]).sort_values(["pixel_id", "date"])
available = sorted(df["pixel_id"].unique())
print("Available pixel_ids:", ", ".join(available))

while True:
    pid = input("\nEnter pixel_id (e.g. 0_1), or press Enter to quit: ").strip()
    if not pid:
        break
    g = df[df["pixel_id"] == pid]
    if g.empty:
        print(f"'{pid}' not found.")
        continue
    g = g[g["aet"] <= 10]
    if g.empty:
        print(f"No values <= 10 for '{pid}'.")
        continue

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(g["date"], g["aet"], marker="o", markersize=3, linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("AET")
    ax.set_title(f"AET vs date, pixel {pid} (x={g['x'].iloc[0]:.3f}, y={g['y'].iloc[0]:.3f})")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()
