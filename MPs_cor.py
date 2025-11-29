import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/major_trace_comb_half_lod.csv")

trace_cols = ['Li','Be','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','As','Se','Sr','Mo','Cd','Sb','Ba','Tl','Pb','U']
major_cols = ['Na','Mg','K','Ca','F','Cl','NO3','PO4','SO4','HCO3']
field_cols = ['pH', 'EC']
mp_col = 'MP_total'

vars_to_test = trace_cols + major_cols + field_cols + [mp_col]
corr = df[vars_to_test].corr(method='spearman')

corr_MP = corr[mp_col].sort_values(ascending=False)
corr_MP


cols_to_test = ['MP_total'] + trace_cols + ['pH','EC','Na','Mg','K','Ca','Cl','SO4','HCO3']

results = []
for var in cols_to_test[1:]:
    rho, p = spearmanr(df['MP_total'], df[var], nan_policy='omit')
    results.append([var, rho, p])

corr_stats = pd.DataFrame(results, columns=['Variable','Spearman_rho','p_value'])
corr_stats['Significant (α=0.05)'] = corr_stats['p_value'] < 0.05
corr_stats.sort_values('Spearman_rho', ascending=False, inplace=True)
corr_stats
#print(corr_stats)

corr_stats['adj_p'] = multipletests(corr_stats['p_value'], method='fdr_bh')[1]
corr_stats['Significant_adj (α=0.05)'] = corr_stats['adj_p'] < 0.05
corr_stats
#print(corr_stats)

sig = corr_stats[corr_stats['Significant_adj (α=0.05)']]
sns.barplot(data=sig, x='Spearman_rho', y='Variable', palette='coolwarm')
plt.axvline(0, color='black')
plt.title('Significant correlations with MP_total')
plt.show()
