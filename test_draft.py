import pandas as pd

df = pd.read_csv('./data/sabdab_summary_all.tsv', sep='\t', engine='python')
print(df.dtypes)
print(df.head())
