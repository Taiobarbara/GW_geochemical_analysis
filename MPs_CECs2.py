import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

# Load data
df = pd.read_csv("/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/MP_size_mass_CEC_ng.csv")

# --- Create CEC source indices ---
mp_bulk = ["MPs_ppl", "MPs_mg"]

polymers = ["PVC", "PE", "PP", "PS", "PFTE", "PA", "PET", "other"]
mp_metrics = ["ppl", "mass", "area"]

mp_poly = [f"{p}_{m}" for p in polymers for m in mp_metrics]

df["plastic_additives"] = df[
    ["BPA", "BTA", "BPS", "ethylparaben", "methylparaben"]
].sum(axis=1)

df["pharma"] = df[
    ["theophylline", "valsartan", "diclofenac", "iohexol", "paracetamol"]
].sum(axis=1)

df["daily_markers"] = df[
    ["caffeine", "saccharin", "cotinine", "paraxanthine"]
].sum(axis=1)

cec_cols = [
    "plastic_additives",
    "pharma",
    "daily_markers"
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

print(filtered)


