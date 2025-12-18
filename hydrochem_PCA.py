import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/MP_size_mass_CEC_ng.csv")

hydro_vars = [
    "pH", "EC", "TDS",
    "Na_meq", "Ca_meq", "Mg_meq",
    "Cl_meq", "SO4_meq", "HCO3_meq"
]

X = df[hydro_vars].copy()

# Standardisation is mandatory
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
scores = pca.fit_transform(X_scaled)

expl_var = pca.explained_variance_ratio_ * 100


pca_df = pd.DataFrame(
    scores,
    columns=["PC1", "PC2"]
)

pca_df["Label"] = df["Label"]
pca_df["Campaign"] = df["Campaign"]

loadings = pd.DataFrame(
    pca.components_.T,
    index=hydro_vars,
    columns=["PC1", "PC2"]
)

plt.figure(figsize=(8, 7))

# Samples
for camp, marker in zip([1, 2], ["o", "s"]):
    subset = pca_df[pca_df["Campaign"] == camp]
    plt.scatter(
        subset["PC1"], subset["PC2"],
        marker=marker, s=70, label=f"Campaign {camp}"
    )

# Loadings
for var in loadings.index:
    plt.arrow(
        0, 0,
        loadings.loc[var, "PC1"] * 3,
        loadings.loc[var, "PC2"] * 3,
        head_width=0.05
    )
    plt.text(
        loadings.loc[var, "PC1"] * 3.2,
        loadings.loc[var, "PC2"] * 3.2,
        var, fontsize=9
    )

plt.axhline(0)
plt.axvline(0)

plt.xlabel(f"PC1 ({expl_var[0]:.1f}%)")
plt.ylabel(f"PC2 ({expl_var[1]:.1f}%)")
plt.legend()
plt.title("Hydrochemical PCA (active variables only)")
plt.tight_layout()
plt.show()

mp_overlay = [
    "PVC_ppl", "PS_ppl", "PFTE_ppl", "PET_ppl", "MPs_ppl"
]

cec_overlay = [
    "BTA", "BPS", "salicylic_acid", "DEET",
    "ensulizole", "caffeine", "valsartan"
]

def correlate_with_pcs(df, vars_to_project, pcs):
    out = {}
    for v in vars_to_project:
        r1, _ = spearmanr(df[v], pcs[:, 0], nan_policy="omit")
        r2, _ = spearmanr(df[v], pcs[:, 1], nan_policy="omit")
        out[v] = [r1, r2]
    return pd.DataFrame(out, index=["PC1", "PC2"]).T

mp_proj = correlate_with_pcs(df, mp_overlay, scores)
cec_proj = correlate_with_pcs(df, cec_overlay, scores)

plt.figure(figsize=(8, 7))

# Samples
for camp, marker in zip([1, 2], ["o", "s"]):
    subset = pca_df[pca_df["Campaign"] == camp]
    plt.scatter(
        subset["PC1"], subset["PC2"],
        marker=marker, s=70
    )

# Hydro loadings
for var in loadings.index:
    plt.arrow(0, 0, loadings.loc[var, "PC1"] * 3,
              loadings.loc[var, "PC2"] * 3, alpha=0.4)
    plt.text(loadings.loc[var, "PC1"] * 3.2,
             loadings.loc[var, "PC2"] * 3.2,
             var, fontsize=8)

# MPs
for v in mp_proj.index:
    plt.arrow(0, 0, mp_proj.loc[v, "PC1"] * 2,
              mp_proj.loc[v, "PC2"] * 2, linestyle="--")
    plt.text(mp_proj.loc[v, "PC1"] * 2.1,
             mp_proj.loc[v, "PC2"] * 2.1,
             v, fontsize=9)

# CECs
for v in cec_proj.index:
    plt.arrow(0, 0, cec_proj.loc[v, "PC1"] * 2,
              cec_proj.loc[v, "PC2"] * 2)
    plt.text(cec_proj.loc[v, "PC1"] * 2.1,
             cec_proj.loc[v, "PC2"] * 2.1,
             v, fontsize=9)

plt.axhline(0)
plt.axvline(0)

plt.xlabel(f"PC1 ({expl_var[0]:.1f}%)")
plt.ylabel(f"PC2 ({expl_var[1]:.1f}%)")
plt.title("Hydrochemical PCA with MPs and CECs projected")
plt.tight_layout()
plt.show()

