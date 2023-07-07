# %%
import pandas as pd
import numpy as np
import boatdata
from tqdm import tqdm


def ret_sanren_onehot():
    sanren_num = boatdata.ret_sanren()

    th123 = np.array(df[['th_1', 'th_2', 'th_3']])
    th123 = th123[:, 0]*100 + th123[:, 1]*10 + th123[:, 2]*1

    sanren_one_hot = 1*((sanren_num - th123.reshape(-1, 1)) == 0)

    return sanren_one_hot

def ret_niren_onehot():
    niren_num = boatdata.ret_niren()

    th12 = np.array(df[['th_1', 'th_2']])
    th12 = th12[:, 0]*10 + th12[:, 1]*1

    niren_one_hot = 1*((niren_num - th12.reshape(-1, 1)) == 0)

    return niren_one_hot

# %%
df = pd.read_csv('datas/boatdata.csv')
# %%
bt = boatdata.BoatDataset(df)
# %%
san = 1/bt.niren_odds
# %%
odds = np.sum(bt.niren_odds*ret_niren_onehot(), axis=1)
# %%
kane = 0
win, lose = 0, 0

n = -70
for z in tqdm(range(1000)):
    n = -1*(z+1)

    sa = (san[:n] - san[n])

    sm = np.sum(sa**2, axis=1)
    indx = np.argsort(sm)

    ka = bt.th[:n][:, :2][indx]
    th = bt.th[:n][:, 0]*10 + bt.th[:n][:, 1]*1

    sax = []
    for i in bt.niren_indx:
        sax.append(np.sum((th==i)*(1/sm)))
    sax = np.array(sax)/len(sax)

    wariai = sax/np.sum(sax)
    kitai = bt.niren_odds[n]*wariai

    bet = bt.niren_indx[(kitai > 3)&(sax > 3000)][:3]

    a = bt.th[:, :2][n]
    a = a[0]*10+a[1]

    if a in bet:
        kane += odds[n] - len(bet)
        win += odds[n]
    else:
        kane += - len(bet)
    lose += len(bet)
# %%
print(kane)
print(win/lose)
# %%
win
# %%
lose
# %%
