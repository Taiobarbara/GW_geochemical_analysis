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
        row["Na_meq"],
        color=row["Color"],
        marker=row["Marker"],
        s=row["Size"],
        alpha=row["Alpha"],
        label=row["Label"]
    )

# === Fix duplicate legend entries ===
handles, labels = plt.gca().get_legend_handles_labels()
unique = dict(zip(labels, handles))
plt.legend(unique.values(), unique.keys(), title="Samples")

# === Axes formatting ===
plt.ylabel("Na (meq/L)")
plt.xlabel("Cl (meq/L)")
plt.title("Gibbs Diagram 2: Cl vs Na")

# === Build reference lines ===
x_line = np.linspace(0, df["Cl_meq"].max() * 1.1, 200)

# lake line: y = x
plt.plot(x_line, x_line, linestyle="--", label="Lake")

# seawater line: y = 0.8x
plt.plot(x_line, 0.8 * x_line, linestyle="--", label="SW")

plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
