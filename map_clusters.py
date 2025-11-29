import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
import os

# ---------------- Load and prepare data ----------------
df = pd.read_csv("/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/sampling_table_with_clusters.csv")  # must include lat, lon, cluster, and trace_cols
trace_cols = ['Li','Be','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','As','Se','Sr','Mo','Cd','Sb','Ba','Tl','Pb','U']

# Z-score scaling to create composite index
scaler = StandardScaler()
df["trace_index"] = scaler.fit_transform(df[trace_cols]).mean(axis=1)

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
gdf = gdf.to_crs(epsg=3857)

# Output folder
os.makedirs("cluster_maps", exist_ok=True)

# ---------------- Interpolation (IDW function) ----------------
def idw_interpolate(x, y, z, xi, yi, power=2):
    dist = np.sqrt((x[:, None] - xi[None, :])**2 + (y[:, None] - yi[None, :])**2)
    dist[dist == 0] = 1e-12
    weights = 1 / dist**power
    return np.sum(weights * z[:, None], axis=0) / np.sum(weights, axis=0)

# ---------------- Loop over clusters ----------------
clusters = sorted(df["cluster"].unique())
xmin, ymin, xmax, ymax = gdf.total_bounds

grid_x, grid_y = np.meshgrid(
    np.linspace(xmin, xmax, 250),
    np.linspace(ymin, ymax, 250)
)

for c in clusters:
    sub = gdf[gdf["cluster"] == c]

    # Perform interpolation
    z_interp = idw_interpolate(
        sub.geometry.x.values,
        sub.geometry.y.values,
        sub["trace_index"].values,
        grid_x.ravel(), grid_y.ravel()
    ).reshape(grid_x.shape)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Heatmap
    hm = ax.imshow(
        z_interp, extent=(xmin, xmax, ymin, ymax),
        origin='lower', cmap="Spectral_r", alpha=0.85
    )

    # Piezometer points
    sub.plot(ax=ax, markersize=60, color='black', edgecolor='white')
    for idx, row in sub.iterrows():
        ax.text(row.geometry.x, row.geometry.y, row["Piezometer"], fontsize=8)

    plt.title(f"Spatial Distribution — Cluster {c}", fontsize=15, weight='bold')
    plt.colorbar(hm, ax=ax, label="Trace Index (Z-score)")
    ctx.add_basemap(ax, source=ctx.providers.OpenTopoMap)

    plt.tight_layout()
    fig.savefig(f"cluster_maps/cluster_{c}_map.png", dpi=300)
    plt.close()

print("✔ All maps saved in folder: cluster_maps")
