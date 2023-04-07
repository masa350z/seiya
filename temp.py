# %%
import boatdata
import numpy as np
import pandas as pd

self = boatdata.BoatDataset()
# %%
n = 0.1

field_data = self.ret_field_conditions()

seps = np.zeros(len(field_data)).astype('str')
seps[seps == '0.0'] = '[SEP]'
seps = seps.reshape(-1, 1)

racers_data = self.ret_racers_data()

half_len = int(len(field_data)*0.6)

fdf = pd.DataFrame(field_data[:half_len].reshape(-1))
fdf = np.array(fdf[~fdf.duplicated()]).reshape(-1).astype('str')

# rdf = pd.DataFrame(racers_data[:half_len].reshape(-1))
# rdf = np.array(rdf[~rdf.duplicated()]).reshape(-1).astype('str')
rdf = self.ret_saiyou_senshu(n, half_len)

self.tokenizer.add_tokens(list(fdf), special_tokens=True)
self.tokenizer.add_tokens(list(rdf), special_tokens=True)
self.tokenizer.add_tokens(['grade_A1', 'grade_A2', 'grade_B1', 'grade_B2'],
                            special_tokens=True)

inp_data = np.concatenate(
    [field_data, seps, racers_data],
    axis=1)
# %%
inp_data[0]
ll = list(inp_data[0])
#ll[-2] = 'num4960'
# %%
self.tokenizer(ll)['input_ids']
# %%

# %%
