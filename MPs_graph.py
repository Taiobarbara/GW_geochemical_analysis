import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Create the dataset
# -----------------------------
data = {
    "Type": ["PVC", "PE", "PP", "PS", "PTFE", "PA", "PET", "other"],
    "S1.1": [2.5, 1.8, 0.3, 0.4, 0.4, 0,   0.8, 0.8],
    "S1.2": [1,   0,   0,   0,   0.2, 0,   0.1, 0.2],
    "S2.1": [0.3, 1.5, 0.3, 0,   0.2, 0.1, 0.1, 0.3],
    "S2.2": [0,   1.6, 0.9, 0,   0,   0,   0,   0],
    "S3.1": [0,   1.7, 0.5, 0,   0,   0,   0,   0.3],
    "S3.2": [0,   1.8, 0.2, 0,   0.1, 0,   0,   0.1],
    "S4.1": [0.4, 0.5, 0.3, 0.1, 0.1, 0,   0.1, 0],
    "S4.2": [0.3, 1.5, 0.8, 0,   0,   0,   0,   0.1],
    "S5.1": [0,   16,  4,   1,   5,   0,   0,   14],
    "S5.2": [0,   6,   0,   0,   0,   0,   0,   3],
}

df = pd.DataFrame(data)

# -----------------------------
# Reshape to long format
# -----------------------------
df_long = df.melt(id_vars="Type", var_name="Sample", value_name="Concentration")

df_long["Piezometer"] = df_long["Sample"].str.extract(r"(S\d)")
df_long["Campaign"] = df_long["Sample"].str.extract(r"\.(\d)")

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))

polymers = df_long["Type"].unique()
colors = plt.cm.tab10.colors  # distinct colors

for polymer, color in zip(polymers, colors):
    subset = df_long[df_long["Type"] == polymer]

    # Campaign 1 – circles
    c1 = subset[subset["Campaign"] == "1"]
    ax.scatter(
        c1["Piezometer"],
        c1["Concentration"],
        marker="o",
        color=color,
        label=f"{polymer} (Campaign 1)"
    )

    # Campaign 2 – squares
    c2 = subset[subset["Campaign"] == "2"]
    ax.scatter(
        c2["Piezometer"],
        c2["Concentration"],
        marker="s",
        color=color,
        label=f"{polymer} (Campaign 2)"
    )

# -----------------------------
# Formatting
# -----------------------------
ax.set_xlabel("Piezometer")
ax.set_ylabel("Concentration (particles per litre)")
ax.set_title("Polymer Concentrations by Piezometer and Sampling Campaign")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
ax.grid(True)

plt.tight_layout()
plt.show()
