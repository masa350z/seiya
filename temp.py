# %%
import numpy as np
import pandas as pd

def ret_sanren():
    """
    3連単の番号リストを返す関数

    Returns:
        ndarray: 3連単の番号リスト
    """
    sanren = []
    for i in range(6):
        for j in range(6):
            for k in range(6):
                c1 = i == j
                c2 = i == k
                c3 = j == k
                if not (c1 or c2 or c3):
                    sanren.append((i+1)*100 + (j+1)*10 + (k+1)*1)

    return np.array(sanren)


def ret_niren():
    """
    2連単の番号リストを返す関数

    Returns:
        ndarray: 2連単の番号リスト
    """
    niren = []
    for i in range(6):
        for j in range(6):
            if not i == j:
                niren.append((i+1)*10 + (j+1)*1)

    return np.array(niren)

# %%
df = pd.read_csv('datas/boatdata.csv')
df
# %%
list(df.columns)
# %%

# %%
df_col = ['nirentan_{}'.format(i) for i in ret_niren()]
odds = df[df_col]
odds = np.array(odds, dtype='float32')

# %%
odds
# %%
