import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv('/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/major_trace_comb_NaN.csv') 

trace_cols = ['Li','Be','V','Cr','Mn','Fe','Co','Ni','Cu', 'Zn', 'As', 'Se', 'Sr', 'Mo', 'Cd', 'Sb', 'Ba', 'Tl', 'Pb', 'U'] 
X = df[trace_cols].copy()

X = np.log1p(X)

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())

X = X.dropna(axis=1, how='all')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95)  # keep enough components to explain 95% variance
scores = pca.fit_transform(X_scaled)
explained_var = np.cumsum(pca.explained_variance_ratio_) * 100

inertia = []
silhouette = []
K_range = range(2, min(6, len(X_scaled)))  # ensures we don't exceed n_samples - 1

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scores[:, :3])  # use first 3 PCs
    inertia.append(kmeans.inertia_)
    
    # Compute silhouette only if valid (>= 2 clusters and < n_samples)
    if len(np.unique(labels)) > 1 and len(np.unique(labels)) < len(X_scaled):
        sil = silhouette_score(scores[:, :3], labels)
        silhouette.append(sil)
    else:
        silhouette.append(np.nan)

# Best K selection
best_k = int(pd.Series(silhouette).idxmax() + 2)  # +2 because K_range starts at 2
print(f"✅ Optimal number of clusters (silhouette): {best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(scores[:, :3])

df_clusters = df.copy()
df_clusters['Cluster'] = labels
cluster_means = df_clusters.groupby('Cluster').mean(numeric_only=True)
print(cluster_means.T)

# Merge cluster labels into your dataframe
df_chem = df.copy()
df_chem['Cluster'] = labels

# Define major-ion and field variables to inspect
majors = ['pH','Na','Mg','K','Ca','F','Cl','NO3','PO4','SO4','HCO3']

# Compute cluster-wise means
cluster_summary = df_chem.groupby('Cluster')[majors].mean()
print("\nCluster-wise mean major-ion composition:\n")
print(cluster_summary.round(2))

# --- Boxplots to visualize hydrochemical differences ---
for param in majors:
    plt.figure(figsize=(5,3))
    sns.boxplot(x='Cluster', y=param, data=df_chem, palette='tab10', legend=False)
    plt.title(f"{param} across clusters")
    plt.tight_layout()
    plt.show()

# -----------------------------------------------
# SAFE correlation heatmap (removes NaN/infinite values)
# -----------------------------------------------
corr_cols = majors + ['Li','Sr','Ba','Mn','Fe','Co','Ni','Cu','Zn','As','Cr','Mo','U']

# Select only columns present in the dataframe and clean
corr_df = df_chem[corr_cols].replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
corr_matrix = corr_df.corr().fillna(0)

sns.clustermap(corr_matrix, cmap='coolwarm', center=0)
plt.suptitle("Correlation between major and trace elements (cleaned)", y=1.02)
plt.show()

# ============================================
# STEP 4 – Cluster Samples (Not Variables)
# ============================================
from scipy.cluster.hierarchy import linkage
import seaborn as sns
import matplotlib.pyplot as plt

# Compute linkage on standardized data
Z_samples = linkage(X_scaled, method='ward')

# Create a clustered heatmap of samples vs variables
sns.clustermap(
    pd.DataFrame(X_scaled, index=df.index, columns=corr_cols),
    method='ward',
    metric='euclidean',
    cmap='RdBu_r',
    center=0,
    figsize=(10, 8)
)
plt.suptitle("Hierarchical Clustering of Groundwater Samples", y=1.02, fontsize=14)
plt.show()


# ============================================
# STEP 5 – PCA Biplot with Element-Cluster Overlay
# ============================================
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Re-run PCA (or reuse your existing PCA results)
pca = PCA(n_components=2)
scores = pca.fit_transform(X_scaled)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Retrieve correlation matrix again
corr = df.corr().abs()

# Assign variable cluster groups manually based on heatmap inspection
# (You can adjust these groupings based on your previous visual)
group1 = ['Na', 'Mg', 'Ca', 'HCO3', 'Sr', 'Ba']
group2 = ['Fe', 'Mn', 'As', 'U', 'Zn']
group3 = ['Cu', 'Mo', 'pH']
group4 = ['Cl', 'SO4', 'NO3']

cluster_groups = {
    'Carbonate/Silicate group': group1,
    'Redox group': group2,
    'pH-Sorption group': group3,
    'Anion group': group4
}

colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple']

# Plot PCA biplot with colored loadings
plt.figure(figsize=(8, 7))
for i, (group, elements) in enumerate(cluster_groups.items()):
    for el in elements:
        if el in df.columns:
            idx = df.columns.get_loc(el)
            plt.arrow(0, 0, loadings[idx, 0], loadings[idx, 1],
                      color=colors[i], alpha=0.7, head_width=0.02, linewidth=1.5)
            plt.text(loadings[idx, 0]*1.1, loadings[idx, 1]*1.1, el,
                     color=colors[i], ha='center', va='center', fontsize=9)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
plt.title("PCA Biplot with Element-Cluster Overlay", fontsize=14)
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(cluster_groups.keys(), loc='best')
plt.tight_layout()
plt.show()
