import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

# --------------------------------------------------
# Helper: lighten a color
# --------------------------------------------------
def lighten_color(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except KeyError:
        c = color
    r, g, b = mcolors.to_rgb(c)
    return (1 - amount*(1-r), 1 - amount*(1-g), 1 - amount*(1-b))

# --------------------------------------------------
# Load data
# --------------------------------------------------
file_path = "/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/MP_size_mass_CEC_ng.csv"
df = pd.read_csv(file_path)

compound = "ensulizole"

# Polymer → marker mapping
polymer_masses = {
    "PET_area": "o",
    "PVC_area": "s",
   # "PS_area": "^"
}

plt.figure(figsize=(9, 6))

# --------------------------------------------------
# Plot
# --------------------------------------------------
for polymer_mass, polymer_marker in polymer_masses.items():

    df_plot = df[(df[compound] > 0) & (df[polymer_mass] > 0)]

    for _, row in df_plot.iterrows():

        # Shade by campaign
        if row["Campaign"] == 1:
            color = lighten_color(row["Color"], amount=0.5)
        else:
            color = row["Color"]

        plt.scatter(
            row[polymer_mass],
            row[compound],
            color=color,
            marker=polymer_marker,
            s=row["Size"],
            alpha=row["Alpha"],
            edgecolor="black"
        )

        plt.text(
            row[polymer_mass],
            row[compound],
            row["Label"],
            fontsize=9,
            ha="right",
            va="bottom"
        )

# --------------------------------------------------
# Polymer legend (marker-based)
# --------------------------------------------------
polymer_legend = [
    Line2D([0], [0], marker="o", linestyle="none", color="black", label="PVC"),
    Line2D([0], [0], marker="s", linestyle="none", color="black", label="PET"),
    Line2D([0], [0], marker="^", linestyle="none", color="black", label="PS"),
]

plt.legend(
    handles=polymer_legend,
    title="Polymer type",
    loc="upper right",
    frameon=True
)

# --------------------------------------------------
# Formatting
# --------------------------------------------------
plt.xlabel("Polymer area (µg²/L)")
plt.ylabel(f"{compound} concentration (ng/L)")
plt.title(f"{compound} vs polymer mass")

plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()