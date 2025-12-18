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

results_cec_hydro = []

for cec in cec_cols:
    for hydro in hydro_cols:
        x = df[cec]
        y = df[hydro]

        if x.nunique() < 3 or y.nunique() < 3:
            continue

        rho, p = spearmanr(x, y, nan_policy="omit")

        results_cec_hydro.append({
            "Group": "CECs",
            "CEC": cec,
            "Hydro_variable": hydro,
            "Spearman_rho": rho,
            "p_value": p
        })

cec_hydro_df = pd.DataFrame(results_cec_hydro)

cec_hydro_df["adj_p"] = multipletests(
    cec_hydro_df["p_value"], method="fdr_bh"
)[1]

cec_hydro_df["Significant_adj"] = cec_hydro_df["adj_p"] < 0.05

cec_hydro_filtered = cec_hydro_df[
    cec_hydro_df["Spearman_rho"].abs() >= 0.6
].sort_values("Spearman_rho", ascending=False)

cec_hydro_filtered
print(cec_hydro_filtered)