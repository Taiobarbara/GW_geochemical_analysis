from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/MP_size_mass_CEC_ng.csv")

# --- major ions ---
hydro_cols = [
    "pH", "EC", "TDS",
    #"Na", "Ca", "Mg",
    #"Cl", "SO4", "HCO3",
    "Na_meq", "Ca_meq", "Mg_meq",
    "Cl_meq", "SO4_meq", "HCO3_meq"
]

# --- MP variables ---
mp_bulk = ["MPs_ppl", "MPs_mg"]

polymers = ["PVC", "PE", "PP", "PS", "PFTE", "PA", "PET", "other"]
metrics = ["ppl", "mass", "area"]

mp_poly = [f"{p}_{m}" for p in polymers for m in metrics]
mp_vars = mp_bulk + mp_poly

results_mp_hydro = []

for mp in mp_vars:
    for hydro in hydro_cols:
        x = df[mp]
        y = df[hydro]

        # Skip constants / mostly zeros
        if x.nunique() < 3 or y.nunique() < 3:
            continue

        rho, p = spearmanr(x, y, nan_policy="omit")

        results_mp_hydro.append({
            "Group": "MPs",
            "MP_variable": mp,
            "Hydro_variable": hydro,
            "Spearman_rho": rho,
            "p_value": p
        })

mp_hydro_df = pd.DataFrame(results_mp_hydro)

mp_hydro_df["adj_p"] = multipletests(
    mp_hydro_df["p_value"], method="fdr_bh"
)[1]

mp_hydro_df["Significant_adj"] = mp_hydro_df["adj_p"] < 0.05

mp_hydro_filtered = mp_hydro_df[
    mp_hydro_df["Spearman_rho"].abs() >= 0.6
].sort_values("Spearman_rho", ascending=False)

mp_hydro_filtered
print(mp_hydro_filtered)
