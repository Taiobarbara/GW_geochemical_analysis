import pandas as pd

# Load the template data
df = pd.read_csv('/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/field data/geochem_analysis/data_WQchart.csv')  

# Show the data
df

#from wqchartpy import triangle_piper
#triangle_piper.plot(df, unit='mg/L', figname='triangle Piper diagram', figformat='jpg')

#from wqchartpy import gaillardet
#gaillardet.plot(df, unit='mg/L', figname='Gaillardet diagram', figformat='jpg')

#from wqchartpy import stiff
#stiff.plot(df, unit='mg/L', figname='Stiff diagram', figformat='jpg')

from wqchartpy import hfed
hfed.plot(df, unit='mg/L', figname='HFE-D diagram', figformat='jpg')