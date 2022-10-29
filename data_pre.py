# %%
import pandas as pd
data=pd.read_csv('data/videos.csv',encoding='utf-8',encoding_errors='ignore')

data.dropna(axis=0, how='any', inplace=True)

print(data.head())

# %%
