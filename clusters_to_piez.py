import pandas as pd

df = pd.read_csv("/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/major_trace_comb_half_lod.csv")
trace_cols = ['Li','Be','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','As','Se','Sr','Mo','Cd','Sb','Ba','Tl','Pb','U']

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

X = StandardScaler().fit_transform(df[trace_cols])
kmeans = KMeans(n_clusters=4, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

df.to_csv("sampling_table_with_clusters.csv", index=False)
