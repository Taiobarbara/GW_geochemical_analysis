import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Load the dataset
# ------------------------------------------------------------------
file_path = "/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/MP_size_mass_CEC_ng.csv"
df = pd.read_csv(file_path)

# ------------------------------------------------------------------
# 2. Select compound and polymer
# ------------------------------------------------------------------
compound = "BPA"
polymer_area = "PVC_mass"

# Keep only detected BTA values and non-zero polymer area
df_plot = df[(df[compound] > 0) & (df[polymer_area] > 0)]

# ------------------------------------------------------------------
# 3. Create the plot
# ------------------------------------------------------------------
plt.figure(figsize=(8, 6))

# Campaign 1 → circles
df_c1 = df_plot[df_plot["Campaign"] == 1]
plt.scatter(
    df_c1[polymer_area],
    df_c1[compound],
    marker="o",
    s=80,
    edgecolor="black",
    label="Campaign 1"
)

# Campaign 2 → squares
df_c2 = df_plot[df_plot["Campaign"] == 2]
plt.scatter(
    df_c2[polymer_area],
    df_c2[compound],
    marker="s",
    s=80,
    edgecolor="black",
    label="Campaign 2"
)

# Label points with piezometer ID
for _, row in df_plot.iterrows():
    plt.text(
        row[polymer_area],
        row[compound],
        row["Label"],
        fontsize=9,
        ha="right",
        va="bottom"
    )

# ------------------------------------------------------------------
# 4. Formatting
# ------------------------------------------------------------------
plt.xlabel("PVC area (µm²)") #surface area (µm²/L) #mass (ng/L)
plt.ylabel("BTA concentration (ng/L)")
plt.title("BTA vs PVC area")

plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# ------------------------------------------------------------------
# 5. Show plot
# ------------------------------------------------------------------
plt.show()
