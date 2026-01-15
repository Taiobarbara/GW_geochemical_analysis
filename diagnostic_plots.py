import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Load the dataset
# ------------------------------------------------------------------
file_path = "/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/MP_size_mass_CEC_ng.csv"
df = pd.read_csv(file_path)

# ------------------------------------------------------------------
# 2. Calculate total MP surface area (µm²/L)
# ------------------------------------------------------------------
area_columns = [col for col in df.columns if col.endswith("_area")]
df["MPs_area_total"] = df[area_columns].sum(axis=1)

# 3. Select compound and remove non-detections
# ------------------------------------------------------------------
compound = "BTA"
df_plot = df[df[compound] > 0]

# ------------------------------------------------------------------
# 4. Create Plot 1A: BTA vs MP surface area
# ------------------------------------------------------------------
plt.figure(figsize=(8, 6))

# Campaign 1 → circles
df_c1 = df_plot[df_plot["Campaign"] == 1]
plt.scatter(
    df_c1["MPs_area_total"],
    df_c1[compound],
    marker="o",
    s=80,
    edgecolor="black",
    label="Campaign 1"
)

# Campaign 2 → squares
df_c2 = df_plot[df_plot["Campaign"] == 2]
plt.scatter(
    df_c2["MPs_area_total"],
    df_c2[compound],
    marker="s",
    s=80,
    edgecolor="black",
    label="Campaign 2"
)

# Label points with piezometer ID
for _, row in df_plot.iterrows():
    plt.text(
        row["MPs_area_total"],
        row[compound],
        row["Label"],
        fontsize=9,
        ha="right",
        va="bottom"
    )

# ------------------------------------------------------------------
# 5. Formatting
# ------------------------------------------------------------------
plt.xlabel("Total microplastic surface area (µm²/L)")
plt.ylabel("BTA concentration (ng/L)")
plt.title("BTA vs total microplastic surface area")

plt.legend(frameon=True)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# ------------------------------------------------------------------
# 6. Show plot
# ------------------------------------------------------------------
plt.show()