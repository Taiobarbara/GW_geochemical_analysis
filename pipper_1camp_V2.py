import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------
# Load and clean data
# -------------------------------------------------------------------
major = pd.read_csv('/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/major_ions_2camp.csv')
field = pd.read_csv('/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/field_parameters_2camp.csv')

# Merge on Piezometer
data = pd.merge(major, field[['Piezometer', 'alkalinity (mg/L) on site']], on='Piezometer', how='left')

# Clean numeric values (convert all columns except Piezometer to float)
for col in data.columns:
    if col != 'Piezometer':
        data[col] = pd.to_numeric(data[col].astype(str).str.replace('<', ''), errors='coerce')

# -------------------------------------------------------------------
# Convert mg/L → meq/L
# -------------------------------------------------------------------
def mgL_to_meqL(mgL, molar_mass, charge):
    return (mgL / molar_mass) * abs(charge)

data['Ca_meq']  = mgL_to_meqL(data['Ca2+'], 40.08, 2)
data['Mg_meq']  = mgL_to_meqL(data['Mg2+'], 24.31, 2)
data['Na_meq']  = mgL_to_meqL(data['Na+'], 22.99, 1)
data['K_meq']   = mgL_to_meqL(data['K+'], 39.10, 1)
data['Cl_meq']  = mgL_to_meqL(data['Cl-'], 35.45, -1)
data['SO4_meq'] = mgL_to_meqL(data['SO42-'], 96.06, -2)
data['HCO3_meq'] = mgL_to_meqL(data['alkalinity (mg/L) on site'], 61.016, -1)

# -------------------------------------------------------------------
# Normalize to percentages
# -------------------------------------------------------------------
cation_sum = data[['Ca_meq','Mg_meq','Na_meq','K_meq']].sum(axis=1)
anion_sum  = data[['Cl_meq','SO4_meq','HCO3_meq']].sum(axis=1)

data['Ca_%'] = (data['Ca_meq'] / cation_sum) * 100
data['Mg_%'] = (data['Mg_meq'] / cation_sum) * 100
data['NaK_%'] = ((data['Na_meq'] + data['K_meq']) / cation_sum) * 100

data['Cl_%']   = (data['Cl_meq'] / anion_sum) * 100
data['SO4_%']  = (data['SO4_meq'] / anion_sum) * 100
data['HCO3_%'] = (data['HCO3_meq'] / anion_sum) * 100

# -------------------------------------------------------------------
# Define helper functions for Piper geometry
# -------------------------------------------------------------------
def cation_coords(row):
    x = 0.5 * (2*row['NaK_%'] + row['Mg_%']) / 100
    y = (np.sqrt(3)/2) * row['Ca_%'] / 100
    return x, y

def anion_coords(row):
    x = 1 + 0.5 * (2*row['Cl_%'] + row['SO4_%']) / 100
    y = (np.sqrt(3)/2) * row['HCO3_%'] / 100
    return x, y

def diamond_coords(cat_x, cat_y, an_x, an_y):
    x = 0.5 * (an_x - cat_x) + 0.5
    y = 0.5 * (cat_y + an_y)
    return x, y

# -------------------------------------------------------------------
# Calculate coordinates
# -------------------------------------------------------------------
points = []
for _, row in data.iterrows():
    cx, cy = cation_coords(row)
    ax, ay = anion_coords(row)
    dx, dy = diamond_coords(cx, cy, ax, ay)
    points.append({'Piezometer': row['Piezometer'], 'cx': cx, 'cy': cy, 'ax': ax, 'ay': ay, 'dx': dx, 'dy': dy})

coords = pd.DataFrame(points)

# -------------------------------------------------------------------
# Plot Piper diagram
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8,8))
plt.axis('equal')
ax.set_axis_off()

# Triangles and diamond outlines
triangle_x = [0, 1, 0.5, 0]
triangle_y = [0, 0, np.sqrt(3)/2, 0]
ax.plot(triangle_x, triangle_y, 'k', lw=1)
ax.plot(np.array(triangle_x)+1, triangle_y, 'k', lw=1)

# Diamond frame
diamond = np.array([[0.5,0],[1.5,0],[1, np.sqrt(3)/2],[0, np.sqrt(3)/2],[0.5,0]])
ax.plot(diamond[:,0], diamond[:,1], 'k', lw=1)

# Plot points
for _, p in coords.iterrows():
    ax.scatter(p['cx'], p['cy'], color='blue', s=60)
    ax.scatter(p['ax'], p['ay'], color='red', s=60)
    ax.scatter(p['dx'], p['dy'], color='green', s=60)
    ax.text(p['dx']+0.02, p['dy'], p['Piezometer'], fontsize=9, color='black')

# Labels
ax.text(0.25, -0.08, 'Cations', ha='center', fontsize=11, fontweight='bold')
ax.text(1.25, -0.08, 'Anions', ha='center', fontsize=11, fontweight='bold')
ax.text(0.75, np.sqrt(3)/2 + 0.05, 'Combined (Diamond Field)', ha='center', fontsize=11, fontweight='bold')

plt.title('Piper Diagram – Groundwater from Massaciuccoli Area (2nd Campaign)', fontsize=13, pad=20)
plt.tight_layout()
plt.show()
