# %%
from glob import glob
import pandas as pd
# %%
datas = glob('datas/*.csv')
# %%
dataframe = pd.DataFrame({})
for i in datas:
    dataframe = pd.concat([dataframe, pd.read_csv(i)])
# %%
dataframe = dataframe.sort_values('race_num')
dataframe = dataframe.reset_index(drop=True)
# %%
dataframe.to_csv('datas/boatdata_temp.csv')
# %%
dataframe
# %%
