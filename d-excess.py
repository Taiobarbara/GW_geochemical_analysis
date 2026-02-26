import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------
df = pd.read_csv("/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/water_isotopes.csv")

# --------------------------------------------------
# 2. Calculate d-excess
# --------------------------------------------------
df["d_excess"] = df["d2H"] - 8.0 * df["d18O"]
print(df)

# --------------------------------------------------
# 3. Plot δ18O vs d-excess
# --------------------------------------------------
plt.figure(figsize=(9, 6))

for _, row in df.iterrows():
    plt.scatter(
        row["d18O"],
        row["d_excess"],
        color=row["Color"],
        marker=row["Marker"],
        s=row["Size"],
        alpha=row["Alpha"]
    )
    plt.text(
        row["d18O"] + 0.03,
        row["d_excess"],
        row["Label"],
        fontsize=10,
        va="center"
    )

# --------------------------------------------------
# 4. Reference line (typical meteoric d-excess)
# --------------------------------------------------
plt.axhline(
    y=10,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="Global meteoric d-excess (~10‰)"
)

# --------------------------------------------------
# 5. Formatting
# --------------------------------------------------
plt.xlabel("$\\delta^{18}$O (‰ VSMOW)", fontsize=12)
plt.ylabel("d-excess (‰)", fontsize=12)
plt.title("$\\delta^{18}$O vs d-excess", fontsize=14)

plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()

plt.show()