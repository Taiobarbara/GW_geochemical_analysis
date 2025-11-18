# ============================================
# Groundwater Trace Element Analysis Workflow
# PCA • Clustering • Correlation • Visualization
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage

# --------------------------------------------
# 1️⃣ Load and preprocess dataset
# --------------------------------------------
file_path = '/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/major_trace_comb_NaN.csv'
df = pd.read_csv(file_path)

# Trace element columns — ensure these actually exist in the CSV
trace_cols = ['Li','Be','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
              'As','Se','Sr','Mo','Cd','Sb','Ba','Tl','Pb','U']

# Keep only existing columns
trace_cols = [c for c in trace_cols if c in df.columns]
print(f"✅ Using {len(trace_cols)} trace elements: {trace_cols}")

# Convert to numeric, coercing text values like "<DL" or "ND" to NaN
df[trace_cols] = df[trace_cols].apply(pd.to_numeric, errors='coerce')

# Replace negative values (which break log1p) with NaN
df[trace_cols] = df[trace_cols].mask(df[trace_cols] < 0)

# Log-transform, replace NaN/inf, and fill remaining with mean
X = np.log1p(df[trace_cols])
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())

# Now scale safely
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✅ Data shape after cleaning: {X_scaled.shape}")
print(f"✅ Any NaNs left? {np.isnan(X_scaled).sum()}")

# --------------------------------------------
# 2️⃣ PCA + Optimal Cluster Detection
# --------------------------------------------
pca = PCA(n_components=0.95)
scores = pca.fit_transform(X_scaled)
explained_var = np.cumsum(pca.explained_variance_ratio_) * 100

inertia, silhouette = [], []
K_range = range(2, min(6, len(X_scaled)))  # safe range

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scores[:, :3])
    inertia.append(kmeans.inertia_)
    silhouette.append(silhouette_score(scores[:, :3], labels))

best_k = K_range[np.argmax(silhouette)]
print(f"✅ Optimal number of clusters (silhouette): {best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(scores[:, :3])

df_clusters = df.copy()
df_clusters['Cluster'] = labels

# --------------------------------------------
# 3️⃣ Cluster-wise Mean Compositions
# --------------------------------------------
majors = ['pH','Na','Mg','K','Ca','F','Cl','NO3','PO4','SO4','HCO3']
majors = [m for m in majors if m in df.columns]

cluster_summary = df_clusters.groupby('Cluster')[majors + trace_cols].mean().round(3)
print("\n📊 Cluster-wise Mean Composition:\n", cluster_summary.T)

# Export results
cluster_summary.to_csv('/Users/bazam/dev/GW_geochemistry/cluster_summary.csv', index=True)
print("💾 Saved cluster summary CSV to /Users/bazam/dev/GW_geochemistry/cluster_summary.csv")

# --------------------------------------------
# 4️⃣ Boxplots for Major Ions
# --------------------------------------------
for param in majors:
    plt.figure(figsize=(5, 3))
    sns.boxplot(x='Cluster', y=param, data=df_clusters, hue='Cluster', palette='tab10', legend=False)
    plt.title(f"{param} across clusters")
    plt.tight_layout()
    plt.show()

# --------------------------------------------
# 5️⃣ Correlation Heatmap
# --------------------------------------------
corr_cols = majors + trace_cols
corr_df = df_clusters[corr_cols].replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
corr_matrix = corr_df.corr().fillna(0)

sns.clustermap(corr_matrix, cmap='coolwarm', center=0, figsize=(10, 8))
plt.suptitle("Correlation between major and trace elements", y=1.02)
plt.show()

# --------------------------------------------
# 6️⃣ Hierarchical Clustering of Samples
# --------------------------------------------
sns.clustermap(
    pd.DataFrame(X_scaled, index=df.index, columns=trace_cols),
    method='ward',
    metric='euclidean',
    cmap='RdBu_r',
    center=0,
    figsize=(10, 8)
)
plt.suptitle("Hierarchical Clustering of Groundwater Samples", y=1.02, fontsize=14)
plt.show()

# --------------------------------------------
# 7️⃣ PCA Biplot (Element-Cluster Overlay)
# --------------------------------------------
pca_2d = PCA(n_components=2)
scores_2d = pca_2d.fit_transform(X_scaled)
loadings = pca_2d.components_.T * np.sqrt(pca_2d.explained_variance_)

plt.figure(figsize=(8, 7))
colors = sns.color_palette("tab10", best_k)

# Plot sample clusters
for i in range(best_k):
    plt.scatter(scores_2d[labels == i, 0], scores_2d[labels == i, 1], label=f'Cluster {i}', alpha=0.7)

# Add variable loadings (arrows)
for i, var in enumerate(trace_cols):
    plt.arrow(0, 0, loadings[i, 0]*2, loadings[i, 1]*2, color='black', alpha=0.5, head_width=0.05)
    plt.text(loadings[i, 0]*2.2, loadings[i, 1]*2.2, var, color='black', ha='center', va='center', fontsize=8)

plt.xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% var.)")
plt.ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% var.)")
plt.title("PCA Biplot with Trace Elements and Clusters")
plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.legend()
plt.tight_layout()
plt.show()
