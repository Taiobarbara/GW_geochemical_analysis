import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

# Load data
df = pd.read_csv("/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/MP_size_mass_CEC_ng.csv")

# --- Define variable groups ---
mp_bulk = ["MPs_ppl", "MPs_mg"]

polymers = ["PVC", "PE", "PP", "PS", "PFTE", "PA", "PET", "other"]
mp_metrics = ["ppl", "mass", "area"]

mp_poly = [f"{p}_{m}" for p in polymers for m in mp_metrics]

cec_cols = [
    # plastic additives
    "BPA", "BTA", "BPS", "ethylparaben", "methylparaben",
    # skin care
    "salicylic_acid", "DEET", "ensulizole",
    # pharmaceuticals
    "theophylline", "valsartan", "diclofenac", "iohexol", "paracetamol",
    # daily markers
    "saccharin", "caffeine", "cotinine", "paraxanthine"
]
results = []

for mp in mp_bulk + mp_poly:
    for cec in cec_cols:
        x = df[mp]
        y = df[cec]

        # Skip if all zeros or constant
        if x.nunique() < 3 or y.nunique() < 3:
            continue

        rho, p = spearmanr(x, y, nan_policy="omit")

        results.append({
            "MP_variable": mp,
            "CEC": cec,
            "Spearman_rho": rho,
            "p_value": p
        })

corr_df = pd.DataFrame(results)

#print(corr_df)

corr_df["adj_p"] = multipletests(
    corr_df["p_value"], method="fdr_bh"
)[1]

corr_df["Significant_adj"] = corr_df["adj_p"] < 0.05

filtered = corr_df[
    (corr_df["Spearman_rho"].abs() >= 0.6)
].sort_values("Spearman_rho", ascending=False)

filtered

#print(filtered)

import matplotlib.pyplot as plt

def plot_mp_vs_multiple_cecs(
    df,
    mp_var,
    cec_list,
    y_log=True,
    xlabel=None
):
    plt.figure(figsize=(7, 5))

    markers = {1: "o", 2: "s"}
    colors = plt.cm.tab10.colors  # clean categorical palette

    for i, cec in enumerate(cec_list):
        for campaign in sorted(df["Campaign"].unique()):
            subset = df[df["Campaign"] == campaign]

            plt.scatter(
                subset[mp_var],
                subset[cec],
                marker=markers[campaign],
                color=colors[i % len(colors)],
                s=70,
                alpha=0.75,
                edgecolor="black",
                label=cec if campaign == 1 else None
            )

    if y_log:
        plt.yscale("log")

    plt.xlabel(xlabel if xlabel else mp_var)
    plt.ylabel("CEC concentration (ng/L)")
    plt.title(f"{mp_var} vs individual CECs")

    plt.legend(
        title="CEC",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False
    )

    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

daily_cecs = ["caffeine", "saccharin", "cotinine", "paraxanthine"]

plot_mp_vs_multiple_cecs(
    df,
    mp_var="PS_ppl",
    cec_list=daily_cecs,
    xlabel="PS (particles/L)"
)

pharma_cecs = [
    "theophylline",
    "valsartan",
    "diclofenac",
    "iohexol",
    "paracetamol"
]

plot_mp_vs_multiple_cecs(
    df,
    mp_var="PFTE_ppl",
    cec_list=pharma_cecs,
    xlabel="PFTE (particles/L)"
)

plot_mp_vs_multiple_cecs(
    df,
    mp_var="PS_area",
    cec_list=daily_cecs,
    xlabel="PS surface area (µm²)"
)

import matplotlib.pyplot as plt

pairs = [
    ("PET_ppl", "valsartan", "PET (particles/L)", "Valsartan (ng/L)"),
    ("PVC_ppl", "valsartan", "PVC (particles/L)", "Valsartan (ng/L)"),
    ("PS_ppl", "caffeine", "PS (particles/L)", "Caffeine (ng/L)"),
    ("PFTE_ppl", "salicylic_acid", "PFTE (particles/L)", "Salicylic acid (ng/L)"),
    ("other_ppl", "BTA", "Other co-polymers (particles/L)", "BTA (ng/L)")
]

markers = {1: "o", 2: "s"}
label_colors = {
    "S1": "red",
    "S2": "green",
    "S3": "blue",
    "S4": "orange",
    "S5": "purple"
}

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
axes = axes.flatten()

for ax, (mp, cec, xlabel, ylabel) in zip(axes, pairs):
    for _, row in df.iterrows():
        ax.scatter(
            row[mp],
            row[cec],
            color=label_colors[row["Label"]],
            marker=markers[row["Campaign"]],
            s=70,
            alpha=0.8,
            edgecolor="black"
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

# Remove empty subplot
fig.delaxes(axes[-1])

# === Custom legends ===
campaign_handles = [
    plt.Line2D([0], [0], marker="o", color="k", linestyle="", label="Campaign 1"),
    plt.Line2D([0], [0], marker="s", color="k", linestyle="", label="Campaign 2")
]

label_handles = [
    plt.Line2D([0], [0], marker="o", color=c, linestyle="", label=l)
    for l, c in label_colors.items()
]

fig.legend(
    handles=campaign_handles + label_handles,
    loc="lower center",
    ncol=4,
    frameon=False
)

plt.suptitle("Relationships between selected polymers (ppl) and individual CECs", y=0.98)
plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.show()

import matplotlib.pyplot as plt

polymers = [
    "PVC_ppl", "PE_ppl", "PP_ppl", "PS_ppl",
    "PFTE_ppl", "PA_ppl", "PET_ppl", "other_ppl"
]

additives = ["BPA", "BTA", "BPS", "ethylparaben", "methylparaben"]

poly_colors = {
    "PVC_ppl": "brown",
    "PE_ppl": "green",
    "PP_ppl": "blue",
    "PS_ppl": "purple",
    "PFTE_ppl": "teal",
    "PA_ppl": "orange",
    "PET_ppl": "red",
    "other_ppl": "black"
}

markers = {1: "o", 2: "s"}

fig, axes = plt.subplots(len(additives), 1, figsize=(10, 14), sharex=True)

for ax, additive in zip(axes, additives):
    for poly in polymers:
        for _, row in df.iterrows():
            ax.scatter(
                row[poly],
                row[additive],
                color=poly_colors[poly],
                marker=markers[row["Campaign"]],
                s=60,
                alpha=0.75,
                edgecolor="black"
            )

    ax.set_ylabel(f"{additive} (ng/L)")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

axes[-1].set_xlabel("Polymer concentration (particles/L)")

# === Legends ===
poly_handles = [
    plt.Line2D([0], [0], marker="o", color=c, linestyle="", label=p.replace("_ppl", ""))
    for p, c in poly_colors.items()
]

campaign_handles = [
    plt.Line2D([0], [0], marker="o", color="k", linestyle="", label="Campaign 1"),
    plt.Line2D([0], [0], marker="s", color="k", linestyle="", label="Campaign 2")
]

fig.legend(
    handles=poly_handles + campaign_handles,
    loc="lower center",
    ncol=5,
    frameon=False
)

plt.suptitle(
    "Polymer particle concentrations (ppl) vs plastic additives",
    y=0.98
)

plt.tight_layout(rect=[0, 0.08, 1, 0.96])
plt.show()
