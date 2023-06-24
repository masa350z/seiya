# %%
from keras.activations import gelu
from keras import layers
import numpy as np
import tensorflow as tf
import boat_transformer as btt
import boatdata
from keras.initializers import he_uniform
from tqdm import tqdm
import logging
import os


# TensorFlowの警告メッセージを非表示にする
logging.getLogger("tensorflow").setLevel(logging.ERROR)


def custom_loss(pred, label):
    loss = tf.keras.losses.CategoricalCrossentropy()(label[0], pred[0])
    loss1 = tf.keras.losses.CategoricalCrossentropy()(label[1][:, 0], pred[1])
    loss2 = tf.keras.losses.CategoricalCrossentropy()(label[1][:, 1], pred[2])
    loss3 = tf.keras.losses.CategoricalCrossentropy()(label[1][:, 2], pred[3])

    loss = loss*loss1*loss2*loss3

    return loss


# %%
def calc_acurracy(prediction, label):
    predicted_indices = tf.argmax(prediction, axis=1)
    true_indices = tf.argmax(label, axis=1)

    correct_predictions = tf.equal(predicted_indices, true_indices)

    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return accuracy


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
        th123_one_hot = tf.one_hot(self.th[:, :3]-1, 6)
        self.th123_one_hot = tf.data.Dataset.from_tensor_slices(th123_one_hot)
        self.dataset_y = tf.data.Dataset.zip((self.dataset_y, self.th123_one_hot))
        self.dataset = tf.data.Dataset.zip((self.dataset_x, self.dataset_y))

    def ret_data_x(self):
        filtered_racer = self.ret_known_racer()
        data_x = tf.data.Dataset.from_tensor_slices((filtered_racer, self.grade, self.incose,
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

    def set_dataset(self, batch_size, train_rate=0.6, val_rate=0.2, shuffle=False):
        dataset_size = tf.data.experimental.cardinality(self.dataset).numpy()
        if shuffle:
            self.dataset = self.dataset.shuffle(dataset_size)

        self.train_size = int(train_rate * dataset_size)
        self.val_size = int(val_rate * dataset_size)

        train_dataset = self.dataset.take(self.train_size)
        val_dataset = self.dataset.skip(self.train_size).take(self.val_size)
        test_dataset = self.dataset.skip(self.train_size + self.val_size)

        self.train_dataset = train_dataset.batch(batch_size)
        self.val_dataset = val_dataset.batch(batch_size)
        self.test_dataset = test_dataset.batch(batch_size)

    def ret_known_racer(self, unknown_rate=0.5, k_std=-2):
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


def reduce_sum_odds_vector(odds_vector, filter, seq_len=120):
    vector = odds_vector*tf.reshape(filter, (1, seq_len, 1))
    vector = tf.reduce_sum(vector, axis=1)

    return vector


class OddsRelatedTranceformer(layers.Layer):
    def __init__(self, indx,
                 num_layer_loops, vector_dims,
                 num_heads, inner_dims):
        super().__init__()

        self.indx = indx

        self.filter01_01 = self.ret_filter(1, 1)
        self.filter01_02 = self.ret_filter(1, 2)
        self.filter01_03 = self.ret_filter(1, 3)

        self.filter02_01 = self.ret_filter(2, 1)
        self.filter02_02 = self.ret_filter(2, 2)
        self.filter02_03 = self.ret_filter(2, 3)

        self.filter03_01 = self.ret_filter(3, 1)
        self.filter03_02 = self.ret_filter(3, 2)
        self.filter03_03 = self.ret_filter(3, 3)

        self.filter04_01 = self.ret_filter(4, 1)
        self.filter04_02 = self.ret_filter(4, 2)
        self.filter04_03 = self.ret_filter(4, 3)

        self.filter05_01 = self.ret_filter(5, 1)
        self.filter05_02 = self.ret_filter(5, 2)
        self.filter05_03 = self.ret_filter(5, 3)

        self.filter06_01 = self.ret_filter(6, 1)
        self.filter06_02 = self.ret_filter(6, 2)
        self.filter06_03 = self.ret_filter(6, 3)

        self.related_racer_transformer = RelatedRacerTransformer(num_layer_loops, vector_dims,
                                                                 num_heads, inner_dims)
        self.related_th_transformer = RelatedThTransformer(num_layer_loops, vector_dims,
                                                           num_heads, inner_dims)

    def ret_filter(self, num, th):
        filter = np.array([str(num) in i[th-1:th] for i in self.indx])
        filter = tf.constant(filter, dtype=tf.float32)

        return filter

    def call(self, odds_vector):
        odds_vector01_01 = reduce_sum_odds_vector(odds_vector, self.filter01_01)
        odds_vector01_02 = reduce_sum_odds_vector(odds_vector, self.filter01_02)
        odds_vector01_03 = reduce_sum_odds_vector(odds_vector, self.filter01_03)

        odds_vector02_01 = reduce_sum_odds_vector(odds_vector, self.filter02_01)
        odds_vector02_02 = reduce_sum_odds_vector(odds_vector, self.filter02_02)
        odds_vector02_03 = reduce_sum_odds_vector(odds_vector, self.filter02_03)

        odds_vector03_01 = reduce_sum_odds_vector(odds_vector, self.filter03_01)
        odds_vector03_02 = reduce_sum_odds_vector(odds_vector, self.filter03_02)
        odds_vector03_03 = reduce_sum_odds_vector(odds_vector, self.filter03_03)

        odds_vector04_01 = reduce_sum_odds_vector(odds_vector, self.filter04_01)
        odds_vector04_02 = reduce_sum_odds_vector(odds_vector, self.filter04_02)
        odds_vector04_03 = reduce_sum_odds_vector(odds_vector, self.filter04_03)

        odds_vector05_01 = reduce_sum_odds_vector(odds_vector, self.filter05_01)
        odds_vector05_02 = reduce_sum_odds_vector(odds_vector, self.filter05_02)
        odds_vector05_03 = reduce_sum_odds_vector(odds_vector, self.filter05_03)

        odds_vector06_01 = reduce_sum_odds_vector(odds_vector, self.filter06_01)
        odds_vector06_02 = reduce_sum_odds_vector(odds_vector, self.filter06_02)
        odds_vector06_03 = reduce_sum_odds_vector(odds_vector, self.filter06_03)

        odds_related_racer01 = layers.concatenate([odds_vector01_01, odds_vector01_02, odds_vector01_03])
        odds_related_racer02 = layers.concatenate([odds_vector02_01, odds_vector02_02, odds_vector02_03])
        odds_related_racer03 = layers.concatenate([odds_vector03_01, odds_vector03_02, odds_vector03_03])
        odds_related_racer04 = layers.concatenate([odds_vector04_01, odds_vector04_02, odds_vector04_03])
        odds_related_racer05 = layers.concatenate([odds_vector05_01, odds_vector05_02, odds_vector05_03])
        odds_related_racer06 = layers.concatenate([odds_vector06_01, odds_vector06_02, odds_vector06_03])

        odds_related_th01 = layers.concatenate([odds_vector01_01, odds_vector02_01, odds_vector03_01, odds_vector04_01, odds_vector05_01, odds_vector06_01])
        odds_related_th02 = layers.concatenate([odds_vector01_02, odds_vector02_02, odds_vector03_02, odds_vector04_02, odds_vector05_02, odds_vector06_02])
        odds_related_th03 = layers.concatenate([odds_vector01_03, odds_vector02_03, odds_vector03_03, odds_vector04_03, odds_vector05_03, odds_vector06_03])

        odds_related_racer = tf.stack([odds_related_racer01, odds_related_racer02, odds_related_racer03,
                                       odds_related_racer04, odds_related_racer05, odds_related_racer06], axis=1)

        odds_related_th = tf.stack([odds_related_th01, odds_related_th02, odds_related_th03], axis=1)

        odds_related_racer = self.related_racer_transformer(odds_related_racer)[:, 1:]

        odds_related_th = self.related_th_transformer(odds_related_th)

        odds_related_th01 = odds_related_th[:, 1]
        odds_related_th02 = odds_related_th[:, 2]
        odds_related_th03 = odds_related_th[:, 3]
        odds_related_th = odds_related_th[:, 0]

        return odds_related_racer, odds_related_th, odds_related_th01, odds_related_th02, odds_related_th03


class RelatedRacerTransformer(btt.TransformerBase):
    def __init__(self, num_layer_loops, vector_dims, num_heads, inner_dims):
        super(RelatedRacerTransformer, self).__init__(num_layer_loops, vector_dims, num_heads, inner_dims)

        self.position_vector = btt.positional_encoding(6,
                                                       vector_dims,
                                                       7,
                                                       trainable=False)
        self.dense01 = layers.Dense(vector_dims,
                                    activation=gelu,
                                    kernel_initializer=he_uniform)
        self.dense02 = layers.Dense(vector_dims,
                                    activation=gelu,
                                    kernel_initializer=he_uniform)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        batch_size = x.shape[0]
        x = self.dense01(x)
        x = self.dense02(x)
        x= self.layernorm1(x)
        x = x + self.position_vector

        x = self.add_cls(x, batch_size)

        for i in range(self.num_layer_loops):
            x += self.enc_layers[i](x)

        return x


class RelatedThTransformer(btt.TransformerBase):
    def __init__(self, num_layer_loops, vector_dims, num_heads, inner_dims):
        super(RelatedThTransformer, self).__init__(num_layer_loops, vector_dims, num_heads, inner_dims)

        self.position_vector = btt.positional_encoding(3,
                                                       vector_dims,
                                                       3,
                                                       trainable=False)

        self.dense01 = layers.Dense(vector_dims,
                                    activation=gelu,
                                    kernel_initializer=he_uniform)
        self.dense02 = layers.Dense(vector_dims,
                                    activation=gelu,
                                    kernel_initializer=he_uniform)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        batch_size = x.shape[0]
        x = self.dense01(x)
        x = self.dense02(x)
        x= self.layernorm1(x)
        x = x + self.position_vector

        x = self.add_cls(x, batch_size)

        for i in range(self.num_layer_loops):
            x += self.enc_layers[i](x)

        return x


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
        self.f_l_encoder = btt.F_L_Encoder(vector_dims)
        self.avest_encoder = btt.aveST_Encoder(vector_dims)
        self.racer_winning_rate_encoder = btt.RacerWinningRateEncoder(vector_dims)
        self.motor_boat_winning_rate_encoder = btt.MotorBoatWinningRateEncoder(vector_dims)
        self.current_info_encoder = btt.CurrentInfoTransformer(num_layer_loops, vector_dims,
                                                               num_heads, inner_dims)
        self.start_tenji_encoder = btt.StartTenjiEncoder(vector_dims)
        self.computer_prediction_encoder = btt.ComputerPredictionTransformer(num_layer_loops, vector_dims,
                                                                             num_heads, inner_dims)
        self.prediction_mark_encoder = btt.PredictionMarkEncoder(vector_dims)
        self.field_encoder = btt.FieldEncoder(vector_dims)
        self.odds_encoder = btt.SanrenTanOddsTransformer(num_layer_loops, vector_dims,
                                                         num_heads, inner_dims)

        self.weight01 = tf.Variable(tf.ones(shape=(8,)),
                                    trainable=True,
                                    name='weight01')
        self.weight02 = tf.Variable(tf.ones(shape=(3,)),
                                    trainable=True,
                                    name='weight02')

        self.output_transformer = OutPutTransformer(num_layer_loops, vector_dims, 
                                                    num_heads, inner_dims)

        self.dense01 = layers.Dense(vector_dims*3,
                                    activation=gelu,
                                    kernel_initializer=he_uniform)
        self.dense02 = layers.Dense(vector_dims*3,
                                    activation=gelu,
                                    kernel_initializer=he_uniform)
        self.dense03 = layers.Dense(vector_dims,
                                    activation=gelu,
                                    kernel_initializer=he_uniform)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.decoder_dense = DecoderDense(decoder_dims)
        self.output_layer = layers.Dense(120, activation='softmax')

    def call(self, x):
        racer, grade, incose, flying, latestart, avest, zenkoku_shouritsu, \
            touchui_shouritsu, motor_shouritsu, boat_shouritsu, ex_no, ex_cose, \
            ex_result, ex_start, start_time, tenji_time, computer_prediction, \
            computer_confidence, prediction_mark, field, wether, wind, \
            tempreture, wind_speed, water_tempreture, water_hight, odds = x

        racer, position_vector = self.racer_encoder([racer, grade, incose])
        f_l = self.f_l_encoder([flying, latestart])
        avest = self.avest_encoder(avest)
        racer_winning_rate = self.racer_winning_rate_encoder([zenkoku_shouritsu, touchui_shouritsu])
        motor_boat_winning_rate = self.motor_boat_winning_rate_encoder([motor_shouritsu, boat_shouritsu])
        current_info = self.current_info_encoder([ex_no, ex_cose, ex_result, ex_start])
        start_tenji = self.start_tenji_encoder([start_time, tenji_time])
        computer_prediction = self.computer_prediction_encoder([computer_prediction, computer_confidence])
        prediction_mark = self.prediction_mark_encoder(prediction_mark)
        field = self.field_encoder([field, wether, wind, tempreture, wind_speed, water_tempreture, water_hight])
        odds = self.odds_encoder(tf.math.sqrt(odds))

        racer = racer[:, 1:]
        current_info = current_info[:, :, 0]
        computer_prediction = computer_prediction[:, 0]
        odds = odds[:, 0]

        x = tf.stack([racer, f_l, avest, racer_winning_rate,
                      motor_boat_winning_rate, current_info,
                      start_tenji, prediction_mark], 1)
        x = x*tf.reshape(layers.Softmax()(self.weight01),
                         (1, 8, 1, 1))

        x = tf.math.reduce_sum(x, axis=1)
        x1 = self.output_transformer(x, position_vector)[:, 0]

        x2 = tf.stack([field, computer_prediction, odds], 1)
        x2 = x2*tf.reshape(layers.Softmax()(self.weight02),
                           (1, 3, 1))
        x2 = layers.Flatten()(x2)

        x2_ = self.dense01(x2)
        x2 = self.layernorm1(x2_ + x2)
        x2_ = self.dense02(x2)
        x2 = self.layernorm2(x2_ + x2)
        x2 = self.dense03(x2)

        x = tf.stack([x1, x2], 1)

        x = self.decoder_dense(layers.Flatten()(x))

        return self.output_layer(x)


class SeiyaLight(tf.keras.Model):
    def __init__(self, num_layer_loops, vector_dims,
                 num_heads, inner_dims):
        super(SeiyaLight, self).__init__()

        self.racer_encoder = btt.RacerTransformer(num_layer_loops, vector_dims,
                                                  num_heads, inner_dims)
        self.odds_encoder = btt.SanrenTanOddsTransformer(num_layer_loops, vector_dims,
                                                         num_heads, inner_dims)

        indx = boatdata.ret_sanren().astype('str')
        self.odds_related_encoder = OddsRelatedTranceformer(indx, num_layer_loops, vector_dims,
                                                            num_heads, inner_dims)
        self.output_transformer = OutPutTransformer(num_layer_loops, vector_dims, 
                                                    num_heads, inner_dims)

        self.decoder_dense = DecoderDense(vector_dims)
        self.decoder_dense01 = DecoderDense(vector_dims)
        self.decoder_dense02 = DecoderDense(vector_dims)
        self.decoder_dense03 = DecoderDense(vector_dims)

        self.output_layer = layers.Dense(120, activation='softmax')
        self.output_layer01 = layers.Dense(6, activation='softmax')
        self.output_layer02 = layers.Dense(6, activation='softmax')
        self.output_layer03 = layers.Dense(6, activation='softmax')

    def call(self, x):
        racer, grade, incose, flying, latestart, avest, zenkoku_shouritsu, \
            touchui_shouritsu, motor_shouritsu, boat_shouritsu, ex_no, ex_cose, \
            ex_result, ex_start, start_time, tenji_time, computer_prediction, \
            computer_confidence, prediction_mark, field, wether, wind, \
            tempreture, wind_speed, water_tempreture, water_hight, odds = x

        racer, position_vector = self.racer_encoder([racer, grade, incose])
        odds = self.odds_encoder(tf.math.sqrt(odds))

        racer = racer[:, 1:]
        odds = odds[:, 1:]

        odds_racer, odds_th, odds1th, odds2th, odds3th = self.odds_related_encoder(odds)

        x = racer + odds_racer

        x = self.output_transformer(x)[:, 0]

        x = layers.concatenate([x, odds_th])
        x1 = layers.concatenate([x, odds1th])
        x2 = layers.concatenate([x, odds2th])
        x3 = layers.concatenate([x, odds3th])

        x1 = self.decoder_dense01(x1)
        x2 = self.decoder_dense01(x2)
        x3 = self.decoder_dense01(x3)

        x = self.decoder_dense(x)

        return self.output_layer(x), self.output_layer01(x1), self.output_layer02(x2), self.output_layer03(x3)


class SeiyaTrainer(SeiyaDataSet):
    def __init__(self, weight_name, k_freeze, race_field=None):
        super().__init__(race_field)

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

    def set_model(self, num_layer_loops, vector_dims,
                  num_heads, inner_dims):
        """
        self.model = Seiya(num_layer_loops,
                           vector_dims,
                           num_heads,
                           inner_dims,
                           decoder_dims)
        """
        self.model = SeiyaLight(num_layer_loops,
                                vector_dims,
                                num_heads,
                                inner_dims)

    def set_optimizer(self, learning_rate,
                      weight_learning_rate,
                      output_learning_rate,
                      no_grad=False):

        self.no_grad = no_grad
        self.learning_rate = learning_rate
        self.weight_learning_rate = weight_learning_rate
        self.output_learning_rate = output_learning_rate

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def train_step(self, data_x, data_y):
        with tf.GradientTape() as tape:
            # loss = tf.keras.losses.CategoricalCrossentropy()(data_y, self.model(data_x))

            loss = custom_loss(self.model(data_x), data_y)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        """
        weight_conditions = tf.map_fn(lambda var: any(param in var.name for param in self.weight_params), 
                                    self.model.trainable_variables)
        output_conditions = tf.map_fn(lambda var: any(param in var.name for param in self.output_params), 
                                    self.model.trainable_variables)
        no_grad_conditions = tf.logical_not(tf.map_fn(lambda grad: isinstance(grad, tf.IndexedSlices), gradients))

        weight_mask = tf.cast(weight_conditions, tf.float32) * (self.weight_learning_rate / self.learning_rate)
        output_mask = tf.cast(output_conditions, tf.float32) * (self.output_learning_rate / self.learning_rate)
        no_grad_mask = tf.cast(self.no_grad, tf.float32) * tf.cast(no_grad_conditions, tf.float32) * 0.

        gradients = gradients * (weight_mask + output_mask + no_grad_mask)
        """

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
                prediction, label = [], []
                for x, y in tqdm(self.val_dataset):
                    prediction.append(self.model(x)[0])
                    label.append(y[0])

                prediction = tf.concat(prediction, axis=0)
                label = tf.concat(label, axis=0)

                val_loss = tf.keras.losses.CategoricalCrossentropy()(label, prediction).numpy()
                val_acc = calc_acurracy(prediction, label).numpy()

                if val_loss < self.temp_val_loss:
                    self.last_epoch = epoch
                    self.freeze = 0
                    self.temp_val_loss = val_loss
                    self.temp_val_acc = val_acc
                    self.temp_weights = self.model.get_weights()

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.best_val_acc = val_acc
                        self.model.save_weights(self.weight_name)
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

    def run_train(self, epochs, repeats, per_batch):
        self.best_val_loss = float('inf')
        self.best_val_acc = 0

        for repeat in range(repeats):
            self.temp_val_loss = float('inf')
            self.temp_val_acc = 0

            for epoch in range(epochs):
                self.run_mono_train(epoch, per_batch, repeat=repeat)


# %%
seiya = SeiyaTrainer('datas/pred_sanren/all', k_freeze=3)
# %%
seiya.set_dataset(batch_size=120*2, shuffle=True)
# %%
num_layer_loops = 1
vector_dims = 128*1
num_heads = 8
inner_dims = vector_dims
decoder_dims = vector_dims*2
seiya.set_model(num_layer_loops, vector_dims,
                num_heads, inner_dims)

per_batch = 100
# %%
seiya.set_optimizer(learning_rate=2e-5,
                    weight_learning_rate=2e-2,
                    output_learning_rate=2e-4,
                    no_grad=False)

seiya.run_train(epochs=10, repeats=100, per_batch=per_batch)

# %%
for data_x, data_y in seiya.val_dataset:
    break
# %%
seiya.model(data_x)
# %%
