import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Load data ===
df = pd.read_csv("/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/Gibbs_plot.csv")

# --- CLEAN STYLE COLUMNS ---
defaults = {
    "Color": "black",
    "Marker": "o",
    "Size": 50,
    "Alpha": 0.8,
    "Label": "sample"
}

for col, default in defaults.items():
    if col in df.columns:
        df[col] = df[col].fillna(default)
        df[col] = df[col].replace("nan", default)
        df[col] = df[col].replace("None", default)
        df[col] = df[col].astype(type(default))

# === Create plot ===
plt.figure(figsize=(8, 6))

for _, row in df.iterrows():
    plt.scatter(
        row["Cl_meq"],
        row["SO4_meq"],
        color=row["Color"],
        marker=row["Marker"],
        s=row["Size"],
        alpha=row["Alpha"],
        label=row["Label"]
    )

# === Fix legend duplicates ===
handles, labels = plt.gca().get_legend_handles_labels()
unique = dict(zip(labels, handles))
plt.legend(unique.values(), unique.keys(), title="Samples")

# === Axes formatting ===
plt.xlabel("Cl (meq/L)")
plt.ylabel("SO₄ (meq/L)")
plt.title("Gibbs Diagram 4: SO₄ vs Cl")

# === Set Y-axis limit to 10 ===
plt.ylim(0, 6)

# === Reference lines ===
max_x = df["Cl_meq"].max() * 1.1
x_line = np.linspace(0, max_x, 200)

plt.plot(x_line, 0.375 * x_line, "--", label="Lake: y = 0.375x")
plt.plot(x_line, 0.1 * x_line, "--", label="Seawater: y = 0.1x")

plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
