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
