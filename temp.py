# %%
from glob import glob
import pandas as pd
from tqdm import tqdm
# %%
datas = glob('datas/*.csv')
datas
# %%
dataframe = pd.DataFrame({})
for i in tqdm(datas):
    dataframe = pd.concat([dataframe, pd.read_csv(i)])
# %%
dataframe = dataframe.sort_values('race_num')
dataframe = dataframe.reset_index(drop=True)
dataframe
# %%
dataframe.to_csv('datas/boatdata_temp.csv')
# %%
d = pd.read_csv('backup/boatdata.csv')
# %%
d[d['index'] == 201703090201]