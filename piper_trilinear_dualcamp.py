"""
piper_trilinear_dualcamp.py

Creates a trilinear Piper diagram with:
- Left triangle: cations (Ca, Mg, Na+K)
- Right triangle: anions (HCO3, SO4, Cl)
- Top diamond: combined facies (above the two triangles)
- Two sampling campaigns shown with different markers/colors
- Sample labels shown on diamond points
- Simple hydrochemical facies classification (for legend)

Usage:
    python piper_trilinear_dualcamp.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# ---------------------------
# Parameters / input files
# ---------------------------
MAJOR1 = 'data/major_ions_1camp.csv'
MAJOR2 = 'data/major_ions_2camp.csv'
FIELD1 = 'data/field_parameters_1camp.csv'
FIELD2 = 'data/field_parameters_2camp.csv'

# Output
OUTFILE = 'piper_trilinear_dualcamp.png'

# ---------------------------
# Utility functions
# ---------------------------
def read_and_merge(major_fp, field_fp):
    major = pd.read_csv(major_fp)
    field = pd.read_csv(field_fp)
    # If field file contains alkalinity under a different name, change here
    key = 'Piezometer'
    merged = pd.merge(major, field[[key, 'alkalinity (mg/L) on site']], on=key, how='left')
    # Clean text values like "<0.013" -> numeric (we'll interpret as half DL later if desired)
    for col in merged.columns:
        if col != key:
            # remove stray commas and whitespace
            merged[col] = merged[col].astype(str).str.replace(',', '').str.strip()
            # optional: replace '<value' with half-value; simple approach: remove '<' then convert to numeric
            merged[col] = merged[col].str.replace(r'^<\s*', '', regex=True)
            merged[col] = pd.to_numeric(merged[col], errors='coerce')
    return merged

def mgL_to_meqL(series_mgL, molar_mass, charge):
    # (mg/L) / (g/mol * 1000 mg/g) * (1000 mmol/mol?) simplified:
    # More simply: (mg/L) / molar_mass (g/mol) * (1 mmol/g?) Actually we use:
    # meq/L = (mg/L) / (molar_mass) * (charge)
    # Because mg/L / (g/mol) gives mmol/L *1000? This standard formula used earlier is fine for relative comparisons.
    # We'll keep the consistent formula used earlier:
    return (series_mgL / molar_mass) * abs(charge)

def normalize_to_100(df, cols):
    s = df[cols].sum(axis=1)
    # avoid division by zero
    s_replaced = s.replace(0, np.nan)
    return df[cols].div(s_replaced, axis=0) * 100

# Ternary coordinate transform for an equilateral triangle with base horizontal and top apex at (0.5, sqrt(3)/2)
sqrt3 = math.sqrt(3)
def ternary_to_cartesian(a, b, c, origin_x=0.0, origin_y=0.0, scale=1.0, orientation='left'):
    """
    a, b, c: percents that sum to 100 (a = top apex species percent, b = right apex percent, c = left apex percent)
    orientation: 'left' or 'right' - to place triangles left or right
    returns cartesian x,y coordinates
    """
    # normalize to fraction
    s = a + b + c
    if np.isclose(s, 0) or np.isnan(s):
        return np.nan, np.nan
    fa, fb, fc = a/s, b/s, c/s
    # Using standard barycentric -> cartesian with triangle vertices:
    # top = (0.5, sqrt3/2), left = (0,0), right = (1,0)
    x = 0.5 * (2*fb + fa)  # mapping formula (works consistently with our earlier mapping)
    y = (sqrt3/2) * fa
    # But the formula above expects fractions scaled by 1 (not 0..100), so scale:
    # adjust to 0..1; we already used a/s so okay
    # orientation offset
    if orientation == 'left':
        ox = origin_x
    else:
        # if right triangle, translate by +offset
        ox = origin_x
    return origin_x + x * scale, origin_y + y * scale

def compute_diamond_point(cat_frac, an_frac, cat_origin=(0,0), an_origin=(0,0), scale=1.0, top_center=(0.75,1.05)):
    """
    Create a single diamond coordinate (for plotting combined composition).
    We'll compute the diamond position as the midpoint (average) between
    the projection of the cation and anion coordinates, then move it upward to the top diamond area.
    top_center is the desired center coordinate for the diamond region.
    """
    # Return mean of the two coordinates and then translate toward top_center.
    cx, cy = cat_frac
    ax, ay = an_frac
    mx, my = 0.5*(cx + ax), 0.5*(cy + ay)
    # Move towards top_center by a small factor to place diamond above triangles
    tx = mx + (top_center[0] - 0.5) * 0.9
    ty = my + (top_center[1] - 0.7) * 0.9
    return tx, ty

# ---------------------------
# Read data for two campaigns
# ---------------------------
camp1 = read_and_merge(MAJOR1, FIELD1)
camp2 = read_and_merge(MAJOR2, FIELD2)

# tag campaign
camp1['campaign'] = 'Campaign 1'
camp2['campaign'] = 'Campaign 2'

# Combine for easier handling
df = pd.concat([camp1, camp2], ignore_index=True)

# ---------------------------
# Convert to meq/L
# (molar masses in g/mol, charges as integers)
# ---------------------------
# Note: adjust column names exactly as in your CSV
df['Ca_meq']  = mgL_to_meqL(df['Ca2+'], 40.08, 2)
df['Mg_meq']  = mgL_to_meqL(df['Mg2+'], 24.31, 2)
df['Na_meq']  = mgL_to_meqL(df['Na+'], 22.99, 1)
df['K_meq']   = mgL_to_meqL(df['K+'], 39.10, 1)

df['Cl_meq']  = mgL_to_meqL(df['Cl-'], 35.45, -1)
df['SO4_meq'] = mgL_to_meqL(df['SO42-'], 96.06, -2)
# alkalinity field is HCO3 in mg/L per your notes
df['HCO3_meq'] = mgL_to_meqL(df['alkalinity (mg/L) on site'], 61.016, -1)

# Combine Na+K for plotting percentages
df['NaK_meq'] = df['Na_meq'] + df['K_meq']

# ---------------------------
# Normalize to percent equivalents (0-100)
# ---------------------------
df[['Ca_%','Mg_%','NaK_%']] = normalize_to_100(df, ['Ca_meq','Mg_meq','NaK_meq'])
df[['HCO3_%','SO4_%','Cl_%']] = normalize_to_100(df, ['HCO3_meq','SO4_meq','Cl_meq'])

# ---------------------------
# Compute cartesian coords for triangles
# We'll place:
# - left cation triangle origin at x=0.0 (scale=1)
# - right anion triangle origin at x=1.6 (shifted right)
# - top diamond centered at (0.8, 1.05)
# ---------------------------
left_origin = (0.0, 0.0)
right_origin = (1.6, 0.0)
triangle_scale = 1.0
top_center = (0.8, 1.05)

coords = []
for idx, row in df.iterrows():
    # cation triangle: a=Ca (top), b=Na+K (right), c=Mg (left)
    a_cat, b_cat, c_cat = row['Ca_%'], row['NaK_%'], row['Mg_%']
    cx, cy = ternary_to_cartesian(a_cat, b_cat, c_cat, origin_x=left_origin[0], origin_y=left_origin[1], scale=triangle_scale, orientation='left')

    # anion triangle: a=HCO3 (top), b=Cl (right), c=SO4 (left)
    a_an, b_an, c_an = row['HCO3_%'], row['Cl_%'], row['SO4_%']
    ax, ay = ternary_to_cartesian(a_an, b_an, c_an, origin_x=right_origin[0], origin_y=right_origin[1], scale=triangle_scale, orientation='right')

    # diamond point (combined)
    dx, dy = compute_diamond_point((cx, cy), (ax, ay), top_center=top_center)

    coords.append({'idx': idx, 'Piezometer': row['Piezometer'], 'campaign': row['campaign'],
                   'cx': cx, 'cy': cy, 'ax': ax, 'ay': ay, 'dx': dx, 'dy': dy,
                   'Ca_%': a_cat, 'Mg_%': c_cat, 'NaK_%': b_cat,
                   'HCO3_%': a_an, 'SO4_%': c_an, 'Cl_%': b_an})
coords_df = pd.DataFrame(coords)

# ---------------------------
# Simple hydrochemical facies classification
# (very basic rule-based)
# ---------------------------
def classify_facies(r):
    # cation dominance
    cation_dom = 'Ca+Mg' if (r['Ca_%'] + r['Mg_%']) > r['NaK_%'] else 'Na+K'
    # anion dominance
    anion_dom = 'HCO3' if r['HCO3_%'] > (r['Cl_%'] + r['SO4_%']) else 'Cl+SO4'
    # combine into a short facies string
    if cation_dom == 'Ca+Mg' and anion_dom == 'HCO3':
        return 'Ca-HCO3'
    if cation_dom == 'Ca+Mg' and anion_dom == 'Cl+SO4':
        return 'Ca-(Cl+SO4)'
    if cation_dom == 'Na+K' and anion_dom == 'HCO3':
        return 'Na-HCO3'
    if cation_dom == 'Na+K' and anion_dom == 'Cl+SO4':
        return 'Na-Cl'
    return f'{cation_dom}-{anion_dom}'

coords_df['facies'] = coords_df.apply(classify_facies, axis=1)

# ---------------------------
# Plotting: triangles, ticks, labels, grids
# ---------------------------
fig, ax = plt.subplots(figsize=(9,10))
ax.set_xlim(-0.1, 2.6)
ax.set_ylim(-0.05, 1.5)
ax.set_aspect('equal')
ax.axis('off')

# Helper to draw triangle grid with ticks 0-100 at 20 increments
def draw_triangle(origin_x, origin_y, scale=1.0, labels=None, title=None, side='cation'):
    # triangle vertices
    top = (origin_x + 0.5*scale, origin_y + (sqrt3/2)*scale)
    left = (origin_x + 0.0, origin_y + 0.0)
    right= (origin_x + 1.0*scale, origin_y + 0.0)
    # frame
    ax.plot([left[0], right[0], top[0], left[0]], [left[1], right[1], top[1], left[1]], 'k', lw=1)
    # grid lines (20,40,60,80)
    for t in [20,40,60,80]:
        f = t/100.0
        # lines parallel to each side:
        # line parallel to base between top and side
        p1 = (left[0] + (top[0]-left[0]) * f, left[1] + (top[1]-left[1]) * f)
        p2 = (right[0] + (top[0]-right[0]) * f, right[1] + (top[1]-right[1]) * f)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='0.7', linewidth=0.7, linestyle='--')
        # line parallel to left side:
        q1 = (top[0] + (right[0]-top[0]) * f, top[1] + (right[1]-top[1]) * f)
        q2 = (left[0] + (right[0]-left[0]) * f, left[1] + (right[1]-left[1]) * f)
        ax.plot([q1[0], q2[0]], [q1[1], q2[1]], color='0.7', linewidth=0.7, linestyle='--')
        # line parallel to right side:
        r1 = (top[0] + (left[0]-top[0]) * f, top[1] + (left[1]-top[1]) * f)
        r2 = (right[0] + (left[0]-right[0]) * f, right[1] + (left[1]-right[1]) * f)
        ax.plot
