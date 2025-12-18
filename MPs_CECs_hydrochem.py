import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

df = pd.read_csv("/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/MP_size_mass_CEC_ng.csv")

mp_bulk = ["MPs_ppl", "MPs_mg"]

polymers = ["PVC", "PE", "PP", "PS", "PFTE", "PA", "PET", "other"]
mp_metrics = ["ppl", "mass", "area"]

mp_vars = mp_bulk + [f"{p}_{m}" for p in polymers for m in mp_metrics]

cec_vars = [
    "BPA", "BTA", "BPS", "ethylparaben", "methylparaben",
    "salicylic_acid", "DEET", "ensulizole",
    "theophylline", "valsartan", "diclofenac", "iohexol", "paracetamol",
    "saccharin", "caffeine", "cotinine", "paraxanthine"
]

hydro_vars = [
    "pH", "EC", "TDS",
    "Na_meq", "Mg_meq", "Ca_meq",
    "Cl_meq", "SO4_meq", "HCO3_meq"
]

def spearman_matrix(df, x_vars, y_vars):
    mat = pd.DataFrame(index=y_vars, columns=x_vars, dtype=float)

    for y in y_vars:
        for x in x_vars:
            if df[x].nunique() < 3 or df[y].nunique() < 3:
                mat.loc[y, x] = np.nan
            else:
                rho, _ = spearmanr(df[x], df[y], nan_policy="omit")
                mat.loc[y, x] = rho

    return mat

mp_hydro_corr = spearman_matrix(df, mp_vars, hydro_vars)
cec_hydro_corr = spearman_matrix(df, cec_vars, hydro_vars)

def plot_heatmap(mat, title, cmap="coolwarm"):
    plt.figure(figsize=(0.5 * mat.shape[1] + 4, 0.4 * mat.shape[0] + 3))
    im = plt.imshow(mat, cmap=cmap, vmin=-1, vmax=1)

    plt.colorbar(im, label="Spearman ρ")

    plt.xticks(range(mat.shape[1]), mat.columns, rotation=90)
    plt.yticks(range(mat.shape[0]), mat.index)

    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_heatmap(
    mp_hydro_corr,
    "Spearman Correlation: MPs vs Hydrochemistry"
)

plot_heatmap(
    cec_hydro_corr,
    "Spearman Correlation: CECs vs Hydrochemistry"
)
