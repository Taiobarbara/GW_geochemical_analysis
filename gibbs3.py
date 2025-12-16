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

# === Compute (Ca + Mg) ===
df["CaMg_meq"] = df["Ca_meq"] + df["Mg_meq"]

# === Create plot ===
plt.figure(figsize=(8, 6))

for _, row in df.iterrows():
    plt.scatter(
        row["HCO3_meq"],
        row["CaMg_meq"],
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
plt.xlabel("HCO₃ (meq/L)")
plt.ylabel("Ca + Mg (meq/L)")
plt.title("Gibbs Diagram 3: (Ca + Mg) vs HCO₃")


# === Reference line y = x ===
max_val = max(df["HCO3_meq"].max(), df["CaMg_meq"].max()) * 1.1
x_line = np.linspace(0, max_val, 200)
plt.plot(x_line, x_line, linestyle="--", label="y = x")

plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()