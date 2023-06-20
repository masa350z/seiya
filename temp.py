# %%
from keras.activations import gelu
from keras import layers
import numpy as np
import tensorflow as tf
import boat_transformer
import boatdata
from keras.initializers import he_uniform
# %%
bt = boatdata.BoatDataset()

entry_no = bt.entry_no

sanren_num = boatdata.ret_sanren()

th123 = np.array(bt.df[['th_1', 'th_2', 'th_3']])
th123 = th123[:, 0]*100 + th123[:, 1]*10 + th123[:, 2]*1

sanren_one_hot = 1*((sanren_num - th123.reshape(-1, 1)) == 0)


# %%

# %%
num_layer_loops = 3
vector_dims = 128
num_heads = 8
inner_dims = 120
model = OddsTransformer(num_layer_loops,
                                      vector_dims,
                                      num_heads,
                                      inner_dims)
# %%
#model = FieldEncoder(120)
# %%

res = model(bt.sanren_odds[:2])
res
# %%
res
# %%
a = model.output_dense(tf.expand_dims(res,2))
# %%
a[0][1]