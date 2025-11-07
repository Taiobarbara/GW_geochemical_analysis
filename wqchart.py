"""
Piper diagram for Massaciuccoli groundwater — using WQChartPy
"""

import pandas as pd
import os
import wqchartpy.triangle_piper as tp

# === CONFIG ===
input_csv = "/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/major_trace_comb.csv"   # path to your new single file
output_csv = "/Users/bazam/dev/GW_geochemistry/combined_for_wqchartpy.csv"
output_fig = "piper_massaciuccoli.png"

# === Ensure output folder exists ===
os.makedirs("data", exist_ok=True)

# === Load your clean dataset ===
df = pd.read_csv(input_csv)

# Expected columns now (based on your example):
# ['Piezometer', 'campaign', 'Na', 'Mg', 'K', 'Ca', 'F', 'Cl', 'NO3', 'PO4', 'SO4', 'HCO3', 'Li', 'Be', ...]
# All in mg/L

# === Select the ions used in the Piper diagram ===
# (HCO3 represents alkalinity; PO4, NO3 etc. are not needed for Piper)
ions = ['Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3']

missing = [c for c in ions if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns for Piper plot: {missing}")

# === Prepare WQChartPy input DataFrame ===
df_piper = pd.DataFrame({
    'SampleID': df['Piezometer'],
    'Na': df['Na'],
    'K': df['K'],
    'Ca': df['Ca'],
    'Mg': df['Mg'],
    'Cl': df['Cl'],
    'SO4': df['SO4'],
    'HCO3': df['HCO3'],
    'campaign': df['campaign']
})

# Add required CO3 column (set to zero if not measured)
df_piper['CO3'] = 0.0

df_piper["Label"] = df_piper["SampleID"]

df_piper.to_csv(output_csv, index=False)
print(f"✅ Cleaned dataset saved: {output_csv}")

# === Generate the Piper diagram ===
print("⏳ Generating Piper diagram...")
tp.plot(df_piper, 'mg/L', output_fig, 'png')
print(f"✅ Piper diagram saved to {output_fig}")