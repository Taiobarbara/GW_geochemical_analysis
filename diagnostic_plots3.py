import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Load the dataset
# ------------------------------------------------------------------
file_path = "/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/MP_size_mass_CEC_ng.csv"
df = pd.read_csv(file_path)

# ------------------------------------------------------------------
# 2. Define group and polymer variable
# ------------------------------------------------------------------
group_name = "Group C – Polar / ionisable compounds"
group_compounds = [
    #"BPA", "BPS" #Group A – Plastic additives
    #"DEET", "ethylparaben", "methylparaben", "cotinine" #Group B — Hydrophobic neutral sorbates
    "diclofenac", "valsartan", "ensulizole", "salicylic_acid", "saccharin", "BTA" #Group C – Polar / ionisable compounds
    #"iohexol", "theophylline", "paraxanthine", "paracetamol", "caffeine" #Group D — Highly polar tracers (weak MP affinity)
]
polymer_var = "PVC_area"   # can change to PVC_area, PET_mass, etc.

# Keep rows where polymer is present
df_plot = df[df[polymer_var] > 0]

# ------------------------------------------------------------------
# 3. Create the plot
# ------------------------------------------------------------------
plt.figure(figsize=(9, 6))

markers = {1: "o", 2: "s"}   # campaign 1 = circle, campaign 2 = square
colors = plt.cm.tab10.colors  # distinct colors for compounds

for i, compound in enumerate(group_compounds):
    
    # Keep only detected concentrations for this compound
    df_c = df_plot[df_plot[compound] > 0]
    
    if df_c.empty:
        continue
    
    for campaign in [1, 2]:
        df_cc = df_c[df_c["Campaign"] == campaign]
        
        plt.scatter(
            df_cc[polymer_var],
            df_cc[compound],
            marker=markers[campaign],
            s=80,
            edgecolor="black",
            color=colors[i],
            label=f"{compound} – Campaign {campaign}"
        )
        
        # Label points with piezometer ID
        for _, row in df_cc.iterrows():
            plt.text(
                row[polymer_var],
                row[compound],
                row["Label"],
                fontsize=9,
                ha="right",
                va="bottom"
            )

# ------------------------------------------------------------------
# 4. Formatting
# ------------------------------------------------------------------
plt.xlabel(f"{polymer_var} (ng/L)")
plt.ylabel("Compound concentration (ng/L)")
plt.title(f"{group_name} vs {polymer_var}")

# Remove duplicate legend entries
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=9)

plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# ------------------------------------------------------------------
# 5. Show plot
# ------------------------------------------------------------------
plt.show()
