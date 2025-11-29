import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import os

# Load data
df = pd.read_csv("/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/sampling_table_with_clusters.csv")

# Variables to map
major_cols = ['Na','Mg','K','Ca','F','Cl','NO3','PO4','SO4','HCO3']
trace_cols = ['Li','Be','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','As','Se','Sr','Mo','Cd','Sb','Ba','Tl','Pb','U']
field_cols = ['pH','EC']

vars_to_plot = field_cols + major_cols + trace_cols

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
gdf = gdf.to_crs(epsg=3857)  # Web mercator to match basemap

# Output folder
out_folder = "spatial_maps"
os.makedirs(out_folder, exist_ok=True)

# Loop over variables
for var in vars_to_plot:
    plt.figure(figsize=(7, 7))

    ax = gdf.plot(
        ax=plt.gca(),
        column=var,
        cmap="Spectral_r",
        markersize=gdf[var] * (300 / gdf[var].max()),  # scaled symbol size
        legend=True,
        alpha=0.8
    )

    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=13)

    # Plot formatting
    plt.title(f"Spatial distribution of {var}", fontsize=14)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(f"{out_folder}/{var}_map.png", dpi=300)
    plt.close()

print("✔ Maps generated and saved to folder:", out_folder)
