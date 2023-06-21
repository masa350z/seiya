# %%
from keras.activations import gelu
from keras import layers
import numpy as np
import tensorflow as tf
import boat_transformer as btt
import boatdata
from keras.initializers import he_uniform
# %%
import logging

# TensorFlowの警告メッセージを非表示にする
logging.getLogger("tensorflow").setLevel(logging.ERROR)


# %%
class SeiyaDataSet(boatdata.BoatDataset):
    def __init__(self, race_field=None):
        super().__init__(race_field)

        self.data_x_numpy = [self.entry_no, self.grade, self.incose,
                             self.flying_latestart[:, :, 0],
                             self.flying_latestart[:, :, 1],
                             self.average_starttime,
                             self.zenkoku_shouritsu, self.touchi_shouritsu,
                             self.motor_shouritsu, self.boat_shouritsu,
                             self.ex_no, self.ex_cose,
                             self.ex_result, self.ex_start,
                             self.start_time, self.tenji_time,
                             self.computer_prediction, self.computer_confidence,
                             self.prediction_mark,
                             self.ar_field, self.wether, self.wind,
                             self.tempreture, self.wind_speed,
                             self.water_tempreture, self.water_hight,
                             self.sanren_odds]
        self.data_y_numpy = self.ret_sanren_onehot()

        self.dataset_x = self.ret_data_x()
        self.dataset_y = tf.data.Dataset.from_tensor_slices(self.data_y_numpy)
        self.dataset = tf.data.Dataset.zip((self.dataset_x, self.dataset_y))

    def ret_data_x(self):
        data_x = tf.data.Dataset.from_tensor_slices((self.entry_no, self.grade, self.incose,
                                                     self.flying_latestart[:, :, 0],
                                                     self.flying_latestart[:, :, 1],
                                                     self.average_starttime,
                                                     self.zenkoku_shouritsu, self.touchi_shouritsu,
                                                     self.motor_shouritsu, self.boat_shouritsu,
                                                     self.ex_no, self.ex_cose,
                                                     self.ex_result, self.ex_start,
                                                     self.start_time, self.tenji_time,
                                                     self.computer_prediction, self.computer_confidence,
                                                     self.prediction_mark,
                                                     self.ar_field, self.wether, self.wind,
                                                     self.tempreture, self.wind_speed,
                                                     self.water_tempreture, self.water_hight,
                                                     self.sanren_odds))

        return data_x

    def ret_sanren_onehot(self):
        sanren_num = boatdata.ret_sanren()

        th123 = np.array(self.df[['th_1', 'th_2', 'th_3']])
        th123 = th123[:, 0]*100 + th123[:, 1]*10 + th123[:, 2]*1

        sanren_one_hot = 1*((sanren_num - th123.reshape(-1, 1)) == 0)

        return sanren_one_hot


class OutPutTransformer(btt.TransformerBase):
    def __init__(self, num_layer_loops, vector_dims, num_heads, inner_dims):
        super(OutPutTransformer, self).__init__(num_layer_loops, vector_dims, num_heads, inner_dims)

    def call(self, x, position_vector):
        batch_size = x.shape[0]
        x = x + position_vector

        x = self.add_cls(x, batch_size)

        for i in range(self.num_layer_loops):
            x += self.enc_layers[i](x)

        return x


class DecoderDense(layers.Layer):
    def __init__(self, feature_dims):
        super(DecoderDense, self).__init__()

        self.dense01 = layers.Dense(feature_dims,
                                    activation=gelu,
                                    kernel_initializer=he_uniform)
        self.dense02 = layers.Dense(feature_dims,
                                    activation=gelu,
                                    kernel_initializer=he_uniform)
        self.dense03 = layers.Dense(feature_dims,
                                    activation=gelu,
                                    kernel_initializer=he_uniform)
        self.dense04 = layers.Dense(feature_dims,
                                    activation=gelu,
                                    kernel_initializer=he_uniform)
        self.dense05 = layers.Dense(feature_dims,
                                    activation=gelu,
                                    kernel_initializer=he_uniform)
        self.dense06 = layers.Dense(feature_dims,
                                    activation=gelu,
                                    kernel_initializer=he_uniform)
        self.dense07 = layers.Dense(feature_dims,
                                    activation=gelu,
                                    kernel_initializer=he_uniform)
        self.dense08 = layers.Dense(feature_dims,
                                    activation=gelu,
                                    kernel_initializer=he_uniform)
        self.dense09 = layers.Dense(feature_dims,
                                    activation=gelu,
                                    kernel_initializer=he_uniform)
        self.dense10 = layers.Dense(feature_dims,
                                    activation=gelu,
                                    kernel_initializer=he_uniform)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm4 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm5 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        x_ = self.dense01(x)
        x_ = self.dense02(x_)
        x = self.layernorm1(x_ + x)
        """
        x_ = self.dense03(x)
        x_ = self.dense04(x_)
        x = self.layernorm2(x_ + x)
        x_ = self.dense05(x)
        x_ = self.dense06(x_)
        x = self.layernorm3(x_ + x)
        x_ = self.dense07(x)
        x_ = self.dense08(x_)
        x = self.layernorm4(x_ + x)
        x_ = self.dense09(x)
        x_ = self.dense10(x_)
        x = self.layernorm5(x_ + x)
        """

        return x


class Seiya(tf.keras.Model):
    def __init__(self, num_layer_loops, vector_dims,
                 num_heads, inner_dims, decoder_dims):
        super(Seiya, self).__init__()

        self.racer_encoder = btt.RacerTransformer(num_layer_loops, vector_dims,
                                                  num_heads, inner_dims)
        self.f_l_avest_encoder = btt.F_L_aveST_Encoder(vector_dims)
        self.racer_winning_rate_encoder = btt.RacerWinningRateEncoder(vector_dims)
        self.motor_boat_winning_rate_encoder = btt.MotorBoatWinningRateEncoder(vector_dims)
        self.current_info_encoder = btt.CurrentInfoTransformer(num_layer_loops, vector_dims,
                                                               num_heads, inner_dims)
        self.start_tenji_encoder = btt.StartTenjiEncoder(vector_dims)
        self.computer_prediction_encoder = btt.ComputerPredictionTransformer(num_layer_loops, vector_dims,
                                                                             num_heads, inner_dims)
        self.prediction_mark_encoder = btt.PredictionMarkEncoder(vector_dims)
        self.field_encoder = btt.FieldEncoder(vector_dims)
        self.odds_encoder = btt.OddsTransformer(num_layer_loops, vector_dims,
                                                num_heads, inner_dims)

        self.weight01 = tf.Variable(tf.random.normal(shape=(7,)), trainable=True)

        self.output_transformer = OutPutTransformer(num_layer_loops, vector_dims, 
                                                    num_heads, inner_dims)

        self.decoder_dense = DecoderDense(decoder_dims)
        self.output_layer = layers.Dense(120, activation='softmax')

    def call(self, x):
        racer, grade, incose, flying, latestart, avest, zenkoku_shouritsu, \
            touchui_shouritsu, motor_shouritsu, boat_shouritsu, ex_no, ex_cose, \
            ex_result, ex_start, start_time, tenji_time, computer_prediction, \
            computer_confidence, prediction_mark, field, wether, wind, \
            tempreture, wind_speed, water_tempreture, water_hight, odds = x

        racer, position_vector = self.racer_encoder([racer, grade, incose])
        f_l_avest = self.f_l_avest_encoder([flying, latestart, avest])
        racer_winning_rate = self.racer_winning_rate_encoder([zenkoku_shouritsu, touchui_shouritsu])
        motor_boat_winning_rate = self.motor_boat_winning_rate_encoder([motor_shouritsu, boat_shouritsu])
        current_info = self.current_info_encoder([ex_no, ex_cose, ex_result, ex_start])
        start_tenji = self.start_tenji_encoder([start_time, tenji_time])
        computer_prediction = self.computer_prediction_encoder([computer_prediction, computer_confidence])
        prediction_mark = self.prediction_mark_encoder(prediction_mark)
        field = self.field_encoder([field, wether, wind, tempreture, wind_speed, water_tempreture, water_hight])
        odds = self.odds_encoder(odds)

        racer = racer[:, 1:]
        current_info = current_info[:, :, 0]
        computer_prediction = computer_prediction[:, 0]
        odds = odds[:, 0]

        x = tf.stack([racer, f_l_avest, racer_winning_rate,
                      motor_boat_winning_rate, current_info,
                      start_tenji, prediction_mark], 1)
        x = x*tf.reshape(layers.Softmax()(self.weight01),
                         (1, 7, 1, 1))
        x = tf.math.reduce_sum(x, axis=1)
        x = self.output_transformer(racer, position_vector)[:, 0]
        x = layers.concatenate([x, field, computer_prediction, odds])

        x = self.decoder_dense(x)

        return self.output_layer(x)

# %%
num_layer_loops = 1
vector_dims = 128*1
num_heads = 8
inner_dims = 120
decoder_dims = vector_dims*4
model = Seiya(num_layer_loops,
              vector_dims,
              num_heads,
              inner_dims,
              decoder_dims)
# %%
bt = SeiyaDataSet()
# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)


def train_step(data_x, data_y):
    with tf.GradientTape() as tape:
        loss = tf.keras.losses.CategoricalCrossentropy()(data_y, model(data_x))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


for epoch in range(10):
    for (batch, (data_x, data_y)) in enumerate(bt.dataset.batch(120*5)):
        loss = train_step(data_x, data_y)

        if batch % 60 == 0:
            print(loss)

# %%
res = []
for x, y in bt.dataset.batch(120):
    res.append(model(x).numpy())
# %%
res = np.concatenate(res)
# %%
mx_bool = (res - np.max(res, axis=1).reshape(-1, 1))==0
# %%
mx_bool
# %%
bet = bt.data_y_numpy*mx_bool
# %%
np.sum(bet)/len(bet)
# %%
b2 = bet.reshape(10, 33994, 120)
# %%
np.sum(np.sum(b2, axis=2), axis=1)/33994
# %%
layers.Softmax()(model.weight01)
# %%
[0.24005401, 0.04454923, 0.00624593,
 0.17947966, 0.3091201 , 0.20049496, 0.02005612],
# %%
