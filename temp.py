# %%
from keras.activations import gelu, relu, sigmoid, softmax
from keras import layers
import numpy as np
import pandas as pd
import tensorflow as tf
import boat_transformer as btt
import boatdata
from keras.initializers import he_uniform
from tqdm import tqdm
import logging
import os
import seaborn as sns


# TensorFlowの警告メッセージを非表示にする
logging.getLogger("tensorflow").setLevel(logging.ERROR)


def custom_loss(pred, label):

    odds_return = pred*label
    odds_return = -1*tf.reduce_mean(odds_return)

    return odds_return


# %%
def calc_acurracy(prediction, label):
    predicted_indices = tf.argmax(prediction, axis=1)
    true_indices = tf.argmax(label, axis=1)

    correct_predictions = tf.equal(predicted_indices, true_indices)

    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return accuracy


class SeiyaDataSet(boatdata.BoatDataset):
    def __init__(self, df, race_field=None):
        super().__init__(df, race_field)
        self.data_index = np.array(pd.read_csv('index.csv')).reshape(-1)
        self.data_y = self.ret_sanren_onehot()

        self.odds_sanren_label = self.sanren_odds*self.data_y - 1
        self.odds_sanren_label = tf.data.Dataset.from_tensor_slices(self.odds_sanren_label[self.data_index].astype('float32'))
        self.dataset_y = tf.data.Dataset.from_tensor_slices(self.data_y[self.data_index])
        self.dataset_y = tf.data.Dataset.zip((self.dataset_y, self.odds_sanren_label))

        self.dataset_x = self.ret_data_x()

        self.dataset = tf.data.Dataset.zip((self.dataset_x, self.dataset_y))

    def ret_data_x(self):
        filtered_racer = self.ret_known_racer()
        data_x = tf.data.Dataset.from_tensor_slices((filtered_racer[self.data_index], self.grade[self.data_index], self.incose[self.data_index],
                                                     self.flying_latestart[self.data_index][:, :, 0],
                                                     self.flying_latestart[self.data_index][:, :, 1],
                                                     self.average_starttime[self.data_index],
                                                     self.zenkoku_shouritsu[self.data_index], self.touchi_shouritsu[self.data_index],
                                                     self.motor_shouritsu[self.data_index], self.boat_shouritsu[self.data_index],
                                                     self.ex_no[self.data_index], self.ex_cose[self.data_index],
                                                     self.ex_result[self.data_index], self.ex_start[self.data_index],
                                                     self.start_time[self.data_index], self.tenji_time[self.data_index],
                                                     ))

        return data_x

    def set_dataset(self, batch_size, train_rate=0.6, val_rate=0.2):
        dataset_size = tf.data.experimental.cardinality(self.dataset).numpy()

        self.train_size = int(train_rate * dataset_size)
        self.val_size = int(val_rate * dataset_size)

        train_dataset = self.dataset.take(self.train_size)
        val_dataset = self.dataset.skip(self.train_size).take(self.val_size)
        test_dataset = self.dataset.skip(self.train_size + self.val_size)

        self.train_dataset = train_dataset.batch(batch_size)
        self.val_dataset = val_dataset.batch(batch_size)
        self.test_dataset = test_dataset.batch(batch_size)

    def ret_known_racer(self, unknown_rate=0.7, k_std=-2):
        known_racers, counts = np.unique(self.entry_no[:-int(len(self.entry_no)*unknown_rate)],
                                         return_counts=True)
        ave, std = np.average(counts), np.std(counts)
        known_racers = known_racers[counts > ave + std*k_std]

        known_bool = self.entry_no - known_racers.reshape(-1, 1, 1)
        known_bool = np.sum(known_bool == 0, axis=0)

        return self.entry_no*known_bool

    def ret_sanren_onehot(self):
        sanren_num = boatdata.ret_sanren()

        th123 = np.array(self.df[['th_1', 'th_2', 'th_3']])
        th123 = th123[:, 0]*100 + th123[:, 1]*10 + th123[:, 2]*1

        sanren_one_hot = 1*((sanren_num - th123.reshape(-1, 1)) == 0)

        return sanren_one_hot

    def ret_niren_onehot(self):
        niren_num = boatdata.ret_niren()

        th12 = np.array(self.df[['th_1', 'th_2']])
        th12 = th12[:, 0]*10 + th12[:, 1]*1

        niren_one_hot = 1*((niren_num - th12.reshape(-1, 1)) == 0)

        return niren_one_hot


class OutPutTransformer(btt.TransformerBase):
    def __init__(self, num_layer_loops, vector_dims, num_heads, inner_dims):
        super(OutPutTransformer, self).__init__(num_layer_loops, vector_dims, num_heads, inner_dims)

        self.position_vector = btt.positional_encoding(6,
                                                       vector_dims,
                                                       7,
                                                       trainable=False)

    def call(self, x):
        batch_size = x.shape[0]
        x = x + self.position_vector

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
        x = self.dense01(x)
        x_ = self.dense02(x)
        x = self.layernorm1(x_ + x)
        x_ = self.dense03(x)
        x_ = self.dense04(x_)
        x = self.layernorm2(x_ + x)
        """
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
    def __init__(self, num_layer_loops,
                 vector_dims, num_heads):
        super(Seiya, self).__init__()

        self.racer_encoder = btt.RacerTransformer(num_layer_loops, vector_dims,
                                                  num_heads, vector_dims)
        #self.f_l_encoder = btt.F_L_Encoder(vector_dims)
        #self.avest_encoder = btt.aveST_Encoder(vector_dims)
        self.racer_winning_rate_encoder = btt.RacerWinningRateEncoder(vector_dims)
        #self.motor_boat_winning_rate_encoder = btt.MotorBoatWinningRateEncoder(vector_dims)
        #self.current_info_encoder = btt.CurrentInfoTransformer(num_layer_loops, vector_dims,num_heads, vector_dims)
        self.start_tenji_encoder = btt.StartTenjiEncoder(vector_dims)
        self.output_transformer = OutPutTransformer(num_layer_loops, vector_dims, 
                                                    num_heads, vector_dims)

        self.decoder_dense = DecoderDense(120)
        self.output_layer = layers.Dense(120, activation='softmax')

    def call(self, x):
        racer, grade, incose, flying, latestart, avest, zenkoku_shouritsu, \
            touchui_shouritsu, motor_shouritsu, boat_shouritsu, ex_no, ex_cose, \
            ex_result, ex_start, start_time, tenji_time = x

        racer, position_vector = self.racer_encoder([racer, grade, incose])
        #f_l = self.f_l_encoder([flying, latestart])
        #avest = self.avest_encoder(avest)
        racer_winning_rate = self.racer_winning_rate_encoder([zenkoku_shouritsu, touchui_shouritsu])
        #motor_boat_winning_rate = self.motor_boat_winning_rate_encoder([motor_shouritsu, boat_shouritsu])
        #current_info = self.current_info_encoder([ex_no, ex_cose, ex_result, ex_start])
        start_tenji = self.start_tenji_encoder([start_time, tenji_time])

        racer = racer[:, 1:]
        #current_info = current_info[:, :, 0]

        x = tf.stack([racer,
                      #f_l,
                      #avest, 
                      racer_winning_rate,
                      #motor_boat_winning_rate,
                      #current_info,
                      start_tenji,
                      ], 1)

        x = tf.math.reduce_sum(x, axis=1)
        x = self.output_transformer(x)[:, 0]
        x = self.decoder_dense(x)
        x = self.output_layer(x)

        return x


class SeiyaTrainer(SeiyaDataSet):
    def __init__(self, df, weight_name, k_freeze, race_field=None):
        super().__init__(df, race_field)

        self.weight_params = ['weight']
        self.output_params = ['decoder_dense']

        os.makedirs(weight_name, exist_ok=True)
        self.weight_name = weight_name + '/best_weights'

        self.k_freeze = k_freeze
        self.freeze = 0
        self.last_epoch = 0

        self.temp_weights = None

        self.best_val_loss = float('inf')
        self.best_val_acc = 0

        self.temp_val_loss = float('inf')
        self.temp_val_acc = 0

        self.temp_val_return = 0
        self.best_val_return = 0

    def set_model(self, num_layer_loops,
                  vector_dims, num_heads):

        self.model = Seiya(num_layer_loops,
                           vector_dims,
                           num_heads)

        self.num_layer_loops = num_layer_loops
        self.vector_dims = vector_dims
        self.num_heads = num_heads
        self.inner_dims = inner_dims

    def set_optimizer(self, learning_rate):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def train_step(self, data_x, data_y):
        with tf.GradientTape() as tape:
            loss = tf.keras.losses.CategoricalCrossentropy()(data_y[0], self.model(data_x))

            #loss = custom_loss(self.model(data_x), data_y[1])

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def model_weights_random_init(self, init_ratio=1e-3):
        weights = self.model.get_weights()

        for i, weight in enumerate(weights):
            if len(weight.shape) == 2:
                rand_mask = np.random.binomial(1, init_ratio, size=weight.shape)
                rand_weights = np.random.randn(*weight.shape) * rand_mask
                weights[i] = weight * (1 - rand_mask) + rand_weights

        self.model.set_weights(weights)

    def run_mono_train(self, epoch, per_batch, repeat=0):
        for (batch, (data_x, data_y)) in enumerate(self.train_dataset):
            self.train_step(data_x, data_y)

            if batch % per_batch == 0 and not batch == 0:
                prediction, label, odds = [], [], []
                for x, y in tqdm(self.val_dataset):
                    prediction.append(self.model(x))
                    label.append(y[0])
                    odds.append(y[1])

                prediction = tf.concat(prediction, axis=0)
                label = tf.concat(label, axis=0)
                odds = tf.concat(odds, axis=0)

                val_acc = calc_acurracy(prediction, label).numpy()
                #odds_return = prediction*odds
                #odds_return = tf.reduce_mean(odds_return)

                #val_loss = -1*odds_return
                val_loss = tf.keras.losses.CategoricalCrossentropy()(label, prediction).numpy()
                #val_loss = custom_loss(prediction, odds+1).numpy()


                if val_loss < self.temp_val_loss:
                    self.last_epoch = epoch
                    self.freeze = 0
                    self.temp_val_loss = val_loss
                    self.temp_val_acc = val_acc
                    #self.temp_val_return = odds_return
                    self.temp_weights = self.model.get_weights()

                    """
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.best_val_acc = val_acc
                        #self.best_val_return = odds_return
                        self.model.save_weights(self.weight_name)
                    """
                else:
                    if self.freeze == 0:
                        self.model.set_weights(self.temp_weights)
                        self.model_weights_random_init(init_ratio=0.0001)
                        self.freeze = self.k_freeze

                print('')
                print(self.weight_name)
                print(f"Repeat : {repeat + 1}")
                print(f"Epoch : {epoch + 1}")
                print(f"Temp valid loss : {self.temp_val_loss}")
                print(f"Best valid loss : {self.best_val_loss}")
                print(f"Temp valid acc : {self.temp_val_acc}")
                print(f"Best valid acc : {self.best_val_acc}")
                #print(f"Temp valid return : {self.temp_val_return}")
                #print(f"Best valid return : {self.best_val_return}")


# %%
df = pd.read_csv('datas/boatdata.csv')
#df = pd.read_csv('datas/boatdata_shuffled.csv')
#df.sample(frac=1).reset_index(drop=True).to_csv('datas/boatdata_shuffled.csv')
#indx = np.arange(len(df))
#np.random.shuffle(indx)
#pd.DataFrame(indx, columns=['index']).to_csv('index.csv', index=False)
# %%
per_batch = 100

repeats = 100
epochs = 25

num_layer_loops = 3
vector_dims = 128*5
num_heads = 8
inner_dims = vector_dims

best_val_loss = float('inf')
best_val_acc = 0
best_val_return = 0


for repeat in range(repeats):
    seiya = SeiyaTrainer(df,
                         'datas/pred_sanren/all_onehot',
                         k_freeze=3, race_field=None)

    seiya.set_dataset(batch_size=120*1)

    seiya.set_model(num_layer_loops, vector_dims,
                    num_heads)

    seiya.set_optimizer(learning_rate=2e-4)
    #seiya.model.load_weights('datas/pred_sanren/all_onehot/best_weights')

    seiya.temp_val_loss = float('inf')
    seiya.temp_val_acc = 0
    seiya.temp_val_return = 0

    seiya.best_val_loss = best_val_loss
    seiya.best_val_acc = best_val_acc
    seiya.best_val_return = best_val_return

    seiya.last_epoch = 0

    for epoch in range(epochs):
        seiya.run_mono_train(epoch, per_batch, repeat=repeat)

        best_val_loss = seiya.best_val_loss
        best_val_acc = seiya.best_val_acc
        best_val_return = seiya.best_val_return

        if epoch - seiya.last_epoch >= 2 or seiya.temp_val_loss == 0:
            break

    seiya.model.set_weights(seiya.temp_weights)
    prediction, label, odds = [], [], []
    for x, y in tqdm(seiya.test_dataset):
        prediction.append(seiya.model(x))
        label.append(y[0])
        odds.append(y[1])

    prediction = tf.concat(prediction, axis=0)
    label = tf.concat(label, axis=0)
    odds = tf.concat(odds, axis=0)
    test_loss = tf.keras.losses.CategoricalCrossentropy()(label, prediction).numpy()
    test_acc = calc_acurracy(prediction, label).numpy()

    if test_loss < seiya.best_val_loss:
        best_val_loss = test_loss
        best_val_acc = test_acc
        seiya.model.save_weights(seiya.weight_name)

"""
# %%
seiya = SeiyaTrainer(df,
                        'datas/pred_sanren/all_onehot',
                        k_freeze=5, race_field=None,
                        shuffle=False)

seiya.set_dataset(batch_size=120)

seiya.set_model(num_layer_loops, vector_dims,
                num_heads)

seiya.set_optimizer(learning_rate=2e-4)

seiya.model.load_weights(seiya.weight_name)


# %%
def ret_prediction_label_odds(dataset):
    prediction, label, odds = [], [], []
    for x, y in tqdm(dataset):
        prediction.append(seiya.model(x))
        label.append(y[0])
        odds.append(y[2])

    prediction = tf.concat(prediction, axis=0).numpy()
    label = tf.concat(label, axis=0).numpy()
    odds = tf.concat(odds, axis=0).numpy()

    return prediction, label, odds


def ret_result_matrix(prediction, kitai, odds):
    ret_lis = []
    for pr in tqdm(range(20)):
        temp_lis = []
        pr = (pr+1)/100
        for kk in range(10, 50):
            kk = kk/10

            bet = (prediction > pr)*(kitai > kk)

            temp_lis.append(np.sum(bet*odds))
        ret_lis.append(temp_lis)

    return np.array(ret_lis)


vl_pred, vl_label, vl_odds = ret_prediction_label_odds(seiya.val_dataset)
te_pred, te_label, te_odds = ret_prediction_label_odds(seiya.test_dataset)

ln = len(seiya.sanren_odds)
vl_sanren_odds = seiya.sanren_odds[seiya.data_index][int(ln*0.6):int(ln*0.8)]
te_sanren_odds = seiya.sanren_odds[seiya.data_index][int(ln*0.8):]

vl_kitai = vl_pred*vl_sanren_odds
te_kitai = te_pred*te_sanren_odds
# %%
vl_matrix = ret_result_matrix(vl_pred, vl_kitai, vl_odds)
te_matrix = ret_result_matrix(te_pred, te_kitai, te_odds)
# %%
vl = 1000
sns.heatmap(vl_matrix, center=0, vmax=vl, vmin=-vl,)
# %%
sns.heatmap(te_matrix, center=0, vmax=vl, vmin=-vl,)
# %%
filt = (vl_matrix > 0)*(te_matrix > 0)
# %%
sns.heatmap(np.array(vl_matrix)*filt, vmax=vl, vmin=0,)
# %%
sns.heatmap(np.array(te_matrix)*filt, vmax=vl, vmin=0,)
# %%
"""