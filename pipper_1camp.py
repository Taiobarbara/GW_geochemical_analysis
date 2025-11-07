import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Load major ion data ===
data = pd.read_csv('/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/major_ions_1camp.csv')

# Convert all numeric columns to float (coerce errors → NaN)
for col in data.columns:
    if col != 'Piezometer':
        data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.replace(r'<(\d+\.\d+)', lambda m: float(m.group(1)) / 2, regex=True)
data = data.apply(pd.to_numeric, errors='coerce')

# Convert mg/L to milliequivalents per liter (meq/L)
# (approximate valence and molar masses)
def mgL_to_meqL(mgL, molar_mass, charge):
    return (mgL / molar_mass) * abs(charge)

# Calculate meq/L
data['Ca_meq'] = mgL_to_meqL(data['Ca2+'], 40.08, 2)
data['Mg_meq'] = mgL_to_meqL(data['Mg2+'], 24.31, 2)
data['Na_meq'] = mgL_to_meqL(data['Na+'], 22.99, 1)
data['K_meq']  = mgL_to_meqL(data['K+'], 39.10, 1)

data['Cl_meq'] = mgL_to_meqL(data['Cl-'], 35.45, -1)
data['SO4_meq'] = mgL_to_meqL(data['SO42-'], 96.06, -2)
data['HCO3_meq'] = mgL_to_meqL(data['alkalinity (mg/L) on site'], 61.016, -1)  # from field data

# Normalize to 100% for cations
cation_sum = data[['Ca_meq','Mg_meq','Na_meq','K_meq']].sum(axis=1)
data['Ca_%'] = (data['Ca_meq'] / cation_sum) * 100
data['Mg_%'] = (data['Mg_meq'] / cation_sum) * 100
data['Na_%'] = (data['Na_meq'] / cation_sum) * 100
data['K_%']  = (data['K_meq']  / cation_sum) * 100

# Normalize to 100% for anions
anion_sum = data[['Cl_meq','SO4_meq','HCO3_meq']].sum(axis=1)
data['Cl_%']   = (data['Cl_meq']   / anion_sum) * 100
data['SO4_%']  = (data['SO4_meq']  / anion_sum) * 100
data['HCO3_%'] = (data['HCO3_meq'] / anion_sum) * 100

# --- Simple Placeholder Piper Plot (triangular projections) ---
plt.scatter(data['Ca_%'], data['Cl_%'], s=80, alpha=0.7)
plt.xlabel('Ca²⁺ (%)')
plt.ylabel('Cl⁻ (%)')
plt.title('Simplified Piper Diagram (Cations vs Anions)')
plt.grid(True)
plt.show()

