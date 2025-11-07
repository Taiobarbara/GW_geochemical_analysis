import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# ---------------------------
# 1️⃣ Load and clean data
# ---------------------------

# Replace these with your real file paths
major_ions_1 = pd.read_csv('/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/major_ions_1camp.csv')
major_ions_2 = pd.read_csv('/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/major_ions_2camp.csv')

# Add a campaign column to distinguish them
major_ions_1["campaign"] = "Campaign 1"
major_ions_2["campaign"] = "Campaign 2"

# Combine both campaigns
data = pd.concat([major_ions_1, major_ions_2], ignore_index=True)

# ---------------------------
# 2️⃣ Convert mg/L → meq/L
# ---------------------------
def mgL_to_meqL(mgL, molar_mass, charge):
    mgL = pd.to_numeric(mgL, errors='coerce')
    return (mgL / molar_mass) * abs(charge)

# Ionic conversion
data["Ca_meq"] = mgL_to_meqL(data["Ca2+"], 40.08, 2)
data["Mg_meq"] = mgL_to_meqL(data["Mg2+"], 24.31, 2)
data["Na_meq"] = mgL_to_meqL(data["Na+"], 22.99, 1)
data["K_meq"]  = mgL_to_meqL(data["K+"], 39.10, 1)
data["Cl_meq"] = mgL_to_meqL(data["Cl-"], 35.45, -1)
data["SO4_meq"] = mgL_to_meqL(data["SO42-"], 96.06, -2)
data["HCO3_meq"] = mgL_to_meqL(data["HCO3-"], 61.016, -1) 

# ---------------------------
# 3️⃣ Convert to % equivalents
# ---------------------------
def normalize(df, cols):
    df_sum = df[cols].sum(axis=1)
    return df[cols].div(df_sum, axis=0) * 100

data[["Ca_%", "Mg_%", "Na_%", "K_%"]] = normalize(data, ["Ca_meq","Mg_meq","Na_meq","K_meq"])
data[["Cl_%", "SO4_%", "HCO3_%"]] = normalize(data, ["Cl_meq","SO4_meq","HCO3_meq"])

# ---------------------------
# 4️⃣ Geometry helpers
# ---------------------------
def ternary_coords(a, b, c):
    """Convert ternary composition to Cartesian coords (a,b,c all in %)."""
    total = a + b + c
    if total == 0:
        return (np.nan, np.nan)
    a, b, c = a/total, b/total, c/total
    x = 0.5 * (2*c + b)
    y = (math.sqrt(3)/2) * b
    return x, y

def diamond_coords(cation_xy, anion_xy):
    """Map left and right triangle compositions into diamond."""
    x = 0.5 * (cation_xy[0] + (1 - anion_xy[0]))
    y = 0.5 * (cation_xy[1] + (anion_xy[1]))
    return x, y

# ---------------------------
# 5️⃣ Plot setup
# ---------------------------
fig, ax = plt.subplots(figsize=(10,6))
ax.set_aspect('equal')
ax.axis('off')

offset = 1.4  # distance between triangles
h = math.sqrt(3)/2

def draw_triangle(origin=(0,0), labels=None):
    """Draw a ternary triangle at origin with optional labels."""
    x0, y0 = origin
    tri = np.array([[x0, y0],
                    [x0+1, y0],
                    [x0+0.5, y0+h],
                    [x0, y0]])
    ax.plot(tri[:,0], tri[:,1], 'k', lw=1)
    for i in np.linspace(0.2, 0.8, 4):
        ax.plot([x0+(i/2), x0+1-(i/2)], [y0+i*h, y0+i*h], '--', color='0.8', lw=0.6)
        ax.plot([x0+i, x0+(0.5+i/2)], [y0, y0+(1-i/2)*h], '--', color='0.8', lw=0.6)
        ax.plot([x0+(1-i), x0+(0.5-i/2)], [y0, y0+(1-i/2)*h], '--', color='0.8', lw=0.6)
    if labels:
        ax.text(x0+0.5, y0+h+0.05, labels[0], ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(x0-0.05, y0-0.05, labels[1], ha='right', va='top', fontsize=10, fontweight='bold')
        ax.text(x0+1.05, y0-0.05, labels[2], ha='left', va='top', fontsize=10, fontweight='bold')

# Draw triangles
draw_triangle((0,0), labels=["Ca", "Mg", "Na + K"])
draw_triangle((offset+1,0), labels=["HCO₃", "SO₄", "Cl"])

# Draw diamond
diamond = np.array([
    [offset, h/2],
    [offset+1, h],
    [offset+2, h/2],
    [offset+1, 0],
    [offset, h/2]
])
ax.plot(diamond[:,0], diamond[:,1], 'k', lw=1)
ax.text(offset+1, h+0.1, "Combined (Diamond)", ha='center', fontsize=10, fontweight='bold')

# ---------------------------
# 6️⃣ Plot samples
# ---------------------------
camp_colors = {'Campaign 1': '#1f77b4', 'Campaign 2': '#ff7f0e'}
camp_markers = {'Campaign 1': 'o', 'Campaign 2': 's'}

for camp in data['campaign'].unique():
    sub = data[data['campaign'] == camp]
    for _, row in sub.iterrows():
        # Cation triangle
        x_c, y_c = ternary_coords(row['Ca_%'], row['Mg_%'], row['Na_%'] + row['K_%'])
        ax.scatter(x_c, y_c, color=camp_colors[camp], marker=camp_markers[camp], edgecolor='k', s=60)
        ax.text(x_c+0.02, y_c+0.02, row['Piezometer'], fontsize=8)

        # Anion triangle
        x_a, y_a = ternary_coords(row['Cl_%'], row['SO4_%'], row['HCO3_%'])
        x_a += offset+1
        ax.scatter(x_a, y_a, color=camp_colors[camp], marker=camp_markers[camp], edgecolor='k', s=60)
        ax.text(x_a+0.02, y_a+0.02, row['Piezometer'], fontsize=8)

        # Diamond (combined facies)
        x_d, y_d = diamond_coords((x_c, y_c), (x_a-(offset+1), y_a))
        x_d += offset
        ax.scatter(x_d, y_d, color=camp_colors[camp], marker=camp_markers[camp], edgecolor='k', s=60)
        ax.text(x_d+0.02, y_d+0.02, row['Piezometer'], fontsize=8)

# ---------------------------
# 7️⃣ Legend & title
# ---------------------------
handles = [plt.Line2D([], [], color=c, marker=m, linestyle='', markersize=8, label=camp)
           for camp, (c, m) in zip(camp_colors.keys(), zip(camp_colors.values(), camp_markers.values()))]
ax.legend(handles=handles, loc='lower right', frameon=False)

plt.title("Piper Trilinear Diagram (Massaciuccoli) — 2 Campaigns", fontsize=13, pad=20)
plt.tight_layout()
plt.savefig("piper_trilinear_massaciuccoli.png", dpi=300)
plt.show()
