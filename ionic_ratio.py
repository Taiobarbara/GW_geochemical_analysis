import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/major_anions_half_lod.csv")  

# === Convert mg/L → meq/L ===
df["Cl_meq"] = df["Cl"] / 35.45
df["HCO3_meq"] = df["HCO3"] / 61
df["SO4_meq"] = df["SO4"] / 48

# === Calculate sum (SO4 + HCO3) ===
df["SO4_HCO3_meq"] = df["SO4_meq"] + df["HCO3_meq"]

# === Plot ===
plt.figure(figsize=(7, 6))
plt.scatter(df["SO4_HCO3_meq"], df["Cl_meq"], s=90, facecolor="skyblue", edgecolor="black")

# Labels & title
plt.xlabel("SO₄ + HCO₃  (meq/L)", fontsize=12)
plt.ylabel("Cl  (meq/L)", fontsize=12)
plt.title("Cl vs (SO₄ + HCO₃) Hydrochemical Diagram", fontsize=14)

# === Same axis scale for correct hydrochemical interpretation ===
xy_min = min(df["SO4_HCO3_meq"].min(), df["Cl_meq"].min())
xy_max = max(df["SO4_HCO3_meq"].max(), df["Cl_meq"].max())
plt.xlim(xy_min, xy_max)
plt.ylim(xy_min, xy_max)
plt.gca().set_aspect("equal", adjustable="box")


# === Label points with piezometer IDs if available ===
if "Piezometer" in df.columns:
    for _, row in df.iterrows():
        plt.text(row["SO4_HCO3_meq"], row["Cl_meq"], row["Piezometer"],
                 fontsize=8, ha="center", va="bottom")

plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()