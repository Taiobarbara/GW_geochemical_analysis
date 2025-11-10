# ==============================================
# PCA + Optimal Cluster Detection for Trace Elements
# Author: Barbara (PhD GW Chemistry)
# ==============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -------------------------------
# 1. Load and prepare dataset
# -------------------------------
df = pd.read_csv('/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/major_trace_comb_NaN.csv') 

# Define metadata vs numeric columns
exclude_cols = ['Piezometer', 'campaign']
trace_cols = [c for c in df.columns if c not in exclude_cols]

# Convert to numeric where possible (forces any non-numeric text -> NaN)
df[trace_cols] = df[trace_cols].apply(pd.to_numeric, errors='coerce')

# Now isolate numeric data
X = df[trace_cols].copy()

# Log-transform to reduce skew (optional)
X = np.log1p(X)

# Replace inf/-inf and NaN with column means
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())

# If any column is still entirely NaN, drop it
X = X.dropna(axis=1, how='all')

print(f"✅ Data cleaned: {X.shape[0]} samples, {X.shape[1]} variables used for PCA")

# Check if any NaNs remain (for debugging)
if X.isna().sum().sum() > 0:
    print("⚠️ Warning: some NaNs remain!")
    print(X.isna().sum()[X.isna().sum() > 0])
else:
    print("✅ No missing values remain.")

# Standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 2. Principal Component Analysis
# -------------------------------
pca = PCA(n_components=0.95)  # keep enough components to explain 95% variance
scores = pca.fit_transform(X_scaled)
explained_var = np.cumsum(pca.explained_variance_ratio_) * 100

plt.figure(figsize=(6,4))
plt.plot(np.arange(1, len(explained_var)+1), explained_var, marker='o')
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('PCA Explained Variance')
plt.grid(True)
plt.show()

print(f"✅ Number of components explaining 95% variance: {pca.n_components_}")

# -------------------------------
# Optimal K detection (Elbow + Silhouette)
# -------------------------------
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

# Plot results
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(list(K_range), inertia, '-o')
ax[0].set_title('Elbow Method')
ax[0].set_xlabel('Number of clusters (k)')
ax[0].set_ylabel('Inertia')

ax[1].plot(list(K_range), silhouette, '-o')
ax[1].set_title('Silhouette Score')
ax[1].set_xlabel('Number of clusters (k)')
ax[1].set_ylabel('Score')
plt.tight_layout()
plt.show()

# Best K selection
best_k = int(pd.Series(silhouette).idxmax() + 2)  # +2 because K_range starts at 2
print(f"✅ Optimal number of clusters (silhouette): {best_k}")

# -------------------------------
# 2D PCA plot with cluster colors
# -------------------------------
import seaborn as sns

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(scores[:, :3])

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=scores[:, 0], y=scores[:, 1],
    hue=labels, palette='tab10', s=100, edgecolor='k'
)
for i, txt in enumerate(df['Piezometer']):
    plt.text(scores[i, 0] + 0.02, scores[i, 1] + 0.02, txt, fontsize=8)

plt.title('PCA (PC1 vs PC2) - Cluster Visualization')
plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}% var)")
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()

df_clusters = df.copy()
df_clusters['Cluster'] = labels
cluster_means = df_clusters.groupby('Cluster').mean(numeric_only=True)
print(cluster_means.T)

sns.heatmap(cluster_means.T, cmap='viridis', annot=True, fmt='.2f')
plt.title('Cluster-wise Trace Element Patterns')
plt.show()

