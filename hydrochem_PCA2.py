import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/MP_size_mass_CEC_ng.csv")

# =========================
# 2. DEFINE VARIABLE GROUPS
# =========================

# Hydrochemical variables for PCA
hydro_vars = [
    "pH", "EC", "TDS",
    "Na_meq", "Ca_meq", "Mg_meq",
    "Cl_meq", "SO4_meq", "HCO3_meq"
]

# MPs to project
mp_vars = [
    "PVC_ppl", "PS_ppl", "PFTE_ppl",
    "PET_ppl", "MPs_ppl"
]

# Key CECs to project
cec_vars = [
    "BTA", "BPS", "salicylic_acid",
    "DEET", "ensulizole", "caffeine",
    "valsartan"
]

# =========================
# 3. PCA ON HYDROCHEMISTRY
# =========================

X = df[hydro_vars].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
scores = pca.fit_transform(X_scaled)

# PCA scores dataframe
pca_df = pd.DataFrame(
    scores,
    columns=["PC1", "PC2"],
    index=X.index
)

pca_df["Label"] = df.loc[X.index, "Label"]
pca_df["Campaign"] = df.loc[X.index, "Campaign"]

# Loadings
loadings = pd.DataFrame(
    pca.components_.T,
    index=hydro_vars,
    columns=["PC1", "PC2"]
)

# =========================
# 4. PROJECT MPs AND CECs
# =========================

def project_variables(df, vars_list):
    Z = scaler.transform(df[hydro_vars])
    scores = pca.transform(Z)
    proj = {}
    for v in vars_list:
        proj[v] = np.corrcoef(df[v], scores[:, 0])[0, 1], \
                  np.corrcoef(df[v], scores[:, 1])[0, 1]
    return pd.DataFrame(proj, index=["PC1", "PC2"]).T

mp_proj  = project_variables(df.loc[X.index], mp_vars)
cec_proj = project_variables(df.loc[X.index], cec_vars)

# =========================
# 5. PLOT
# =========================

plt.figure(figsize=(10, 8))

# ---- Sample points ----
markers = {1: "o", 2: "s"}

for camp in [1, 2]:
    subset = pca_df[pca_df["Campaign"] == camp]
    plt.scatter(
        subset["PC1"], subset["PC2"],
        marker=markers[camp],
        s=90,
        alpha=0.8,
        label=f"Campaign {camp}"
    )

# ---- Sample labels ----
for _, row in pca_df.iterrows():
    plt.text(
        row["PC1"] + 0.05,
        row["PC2"] + 0.05,
        row["Label"],
        fontsize=9,
        alpha=0.7
    )

# ---- Hydrochemical vectors (light grey) ----
for var in loadings.index:
    x, y = loadings.loc[var, "PC1"], loadings.loc[var, "PC2"]
    plt.arrow(0, 0, x * 3, y * 3,
              color="grey", alpha=0.35,
              linewidth=1, head_width=0.03)
    plt.text(x * 3.1, y * 3.1, var,
             fontsize=9, color="grey")


# ---- MPs vectors (dashed) ----
for var in mp_proj.index:
   x, y = mp_proj.loc[var, "PC1"], mp_proj.loc[var, "PC2"]
   plt.arrow(0, 0, x * 2, y * 2,
             linestyle="--", linewidth=1.5,
             color="black", head_width=0.04)
   plt.text(x * 2.4, y * 2.5, var,
            fontsize=10)


# ---- CEC vectors (solid, thicker) ----
for var in cec_proj.index:
   x, y = cec_proj.loc[var, "PC1"], cec_proj.loc[var, "PC2"]
   plt.arrow(0, 0, x * 2, y * 2,
             linewidth=1.5,
             color="black", head_width=0.04)
   plt.text(x * 2.8, y * 2.7, var,
            fontsize=10, fontweight="bold")


# ---- Axes & formatting ----
plt.axhline(0, color="steelblue", linewidth=1)
plt.axvline(0, color="steelblue", linewidth=1)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

plt.title("Hydrochemical PCA with MPs and CECs projected")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
