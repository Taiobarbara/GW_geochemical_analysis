import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------
# If your file is named water_isotopes.csv
df = pd.read_csv("/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/water_isotopes.csv")

# --------------------------------------------------
# 2. Define meteoric water lines
# --------------------------------------------------
x_min = df["d18O"].min() - 1
x_max = df["d18O"].max() + 1
x = np.linspace(x_min, x_max, 200)

# Global Meteoric Water Line (Craig, 1961)
gmwl_y = 8.0 * x + 10.0

# Local Meteoric Water Line (Lake Massaciuccoli)
lmwl_y = 7.60 * x + 7.28

# --------------------------------------------------
# 3. Create the plot
# --------------------------------------------------
plt.figure(figsize=(10, 6))

# Plot GMWL
plt.plot(
    x, gmwl_y,
    color="black",
    linestyle="--",
    linewidth=2,
    label="GMWL ($\\delta D = 8\\,\\delta^{18}O + 10$)"
)

# Plot LMWL
plt.plot(
    x, lmwl_y,
    color="black",
    linestyle="-",
    linewidth=2,
    label="LMWL Massaciuccoli ($\\delta D = 7.60\\,\\delta^{18}O + 7.28$)"
)

# --------------------------------------------------
# 4. Plot samples with error bars
# --------------------------------------------------
for _, row in df.iterrows():
    plt.errorbar(
        row["d18O"], row["d2H"],
        xerr=row["st.dev 18O"],
        yerr=row["st.dev 2H"],
        fmt=row["Marker"],
        color=row["Color"],
        markersize=np.sqrt(row["Size"]),
        alpha=row["Alpha"],
        capsize=2
    )

# --------------------------------------------------
# 5. Formatting
# --------------------------------------------------
plt.title("δ²H vs. δ¹⁸O Diagram", fontsize=14)
plt.xlabel("$\\delta^{18}$O (‰ VSMOW)", fontsize=12)
plt.ylabel("$\\delta$D (‰ VSMOW)", fontsize=12)

plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()

# --------------------------------------------------
# 6. Show plot
# --------------------------------------------------
plt.show()

