import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/major_trace_comb_half_lod.csv")
# ensure numeric
num = df.columns.drop(['Piezometer','campaign']) 
df[num] = df[num].apply(pd.to_numeric, errors='coerce')

# detection frequency for trace elements (fraction > 0 or > small threshold)
trace_cols = ['Li','Be','V', 'Cr','Mn','Fe','Co','Ni','Cu','Zn','As','Se','Sr','Mo','Cd','Sb','Ba','Tl','Pb','U']
trace_cols = [c for c in trace_cols if c in df.columns]
det_freq = (df[trace_cols] > 0).sum() / len(df) * 100
#print(det_freq.sort_values(ascending=False))


df_trace = df[trace_cols].copy()

# log-transform (replace zeros / non-detects to avoid log10(0) errors)
df_trace_log = np.log10(df_trace.replace(0, np.nan))

plt.figure(figsize=(18, 6))
sns.boxplot(data=df_trace_log, palette="viridis")
plt.xticks(rotation=60)
plt.ylabel("log₁₀ concentration (mg/L)")
plt.title("Distribution of Trace Elements (log scale)")
plt.tight_layout()
plt.show()


corr_cols = ['pH','EC','Na','Mg','K','Ca','Cl','SO4','HCO3'] + trace_cols
corr_df = df[corr_cols].corr(method='spearman')
sns.clustermap(corr_df, cmap='coolwarm', center=0, figsize=(10,10))
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X_log = np.log1p(df[trace_cols])
X_log = X_log.fillna(X_log.mean())   # if any
X_scaled = StandardScaler().fit_transform(X_log)

pca = PCA(n_components=0.95)   # keep 95% variance
scores = pca.fit_transform(X_scaled)
loadings = pd.DataFrame(pca.components_.T, index=trace_cols, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
print(loadings.round(3))

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# choose scores[:, :n] (e.g. first 3 PCs)
scores_n = scores[:, :3]
# run silhouette loop (2..min(6,n-1))
# final clustering:
kmeans = KMeans(n_clusters=4, random_state=42).fit(scores_n)
df['cluster'] = kmeans.labels_
# cluster means
cluster_means = df.groupby('cluster')[trace_cols + ['pH','EC','Na','Cl','HCO3']].mean()
cluster_means.to_csv("cluster_means_trace_major.csv")


sns.boxplot(x='cluster', y='MP_total', data=df); plt.show()
# correlation
df[['MP_total'] + trace_cols].corr()['MP_total'].sort_values(ascending=False)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# ---- SETTINGS ----
trace_cols = ['Li','Be','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','As','Se','Sr','Mo','Cd','Sb','Ba','Tl','Pb','U']

# df must contain cluster labels already
cluster_means = df.groupby('cluster')[trace_cols].mean()

# ---- LOG10 SCALING (avoid issues with zeros) ----
log_means = np.log10(cluster_means.replace(0, np.nan))
log_means = log_means.fillna(log_means.min().min())   # convert NaN back to lowest log value

# ---- Radar Plot Preparation ----
categories = trace_cols
N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the circle

# ---- Plot ----
plt.figure(figsize=(10, 10))
cmap = cm.get_cmap('Spectral', len(log_means))

for i, (cluster, row) in enumerate(log_means.iterrows()):
    values = row.tolist()
    values += values[:1]
    ax = plt.subplot(2, 2, i + 1, polar=True)
    ax.plot(angles, values, linewidth=2, color=cmap(i), label=f'Cluster {cluster}')
    ax.fill(angles, values, alpha=0.25, color=cmap(i))
    ax.set_title(f'Cluster {cluster}', fontsize=13, fontweight='bold')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8, rotation=45)
    ax.set_yticklabels([])

plt.suptitle('Trace Element Signature per Cluster (log scale)', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

