import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch
from sklearn.discriminant_analysis import StandardScaler

df = pd.read_csv('/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/major_trace_comb_NaN.csv') 
trace_cols = ['Li','Be','V','Cr','Mn','Fe','Co','Ni','Cu', 'Zn', 'As', 'Se', 'Sr', 'Mo', 'Cd', 'Sb', 'Ba', 'Tl', 'Pb', 'U'] 
# numeric & zero handling
for c in trace_cols: df[c] = pd.to_numeric(df[c], errors='coerce')
# detection freq
print((df[trace_cols].notna().sum()/len(df)*100).sort_values(ascending=False))


# log transform
X = df[trace_cols].replace(0,np.nan)
min_nonzero = X[X>0].min().min()
X = X.fillna(min_nonzero/2)
Xlog = np.log10(X)

# boxplot
plt.figure(figsize=(10,5)); sns.boxplot(data=Xlog); plt.xticks(rotation=90); plt.title('trace box'); plt.show()

# correlation
plt.figure(figsize=(10,8)); sns.heatmap(df[trace_cols].corr(method='spearman'), cmap='coolwarm', vmin=-1, vmax=1); plt.show()

# PCA
Xscaled = StandardScaler().fit_transform(Xlog)
pca = PCA(n_components=3); scores = pca.fit_transform(Xscaled)
load = pd.DataFrame(pca.components_.T, index=trace_cols, columns=['PC1','PC2','PC3'])
print(load)

# biplot scores
plt.figure(figsize=(8,6)); sns.scatterplot(x=scores[:,0], y=scores[:,1], hue=df['campaign']); plt.title('PCA scores'); plt.show()

# clustering dendrogram
d = sch.linkage(Xscaled, method='ward'); sch.dendrogram(d, labels=df['Piezometer'].values); plt.show()