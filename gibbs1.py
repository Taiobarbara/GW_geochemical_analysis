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


# === Compute Na/(Na+Ca) ===
df["Na_ratio"] = df["Na_meq"] / (df["Na_meq"] + df["Ca_meq"])

# === Create plot ===
plt.figure(figsize=(8, 6))

for _, row in df.iterrows():
    plt.scatter(
        row["Na_ratio"],
        row["TDS"],
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
plt.yscale("log")

# ---- ADD Y-AXIS TICKS ----
yticks = [1, 10, 100, 1000, 10000]
plt.yticks(yticks, yticks)

plt.xlabel("Na / (Na + Ca) (meq/meq)")
plt.ylabel("TDS (mg/L)")
plt.title("Gibbs Diagram 1: TDS vs Na/(Na+Ca)")

plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()