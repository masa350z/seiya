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
                  num_heads, inner_dims, decoder_dims):

        self.model = Seiya(num_layer_loops,
                           vector_dims,
                           num_heads,
                           inner_dims,
                           decoder_dims)

    def set_optimizer(self, learning_rate,
                      weight_learning_rate,
                      output_learning_rate,
                      no_grad=False):

        self.no_grad = no_grad
        self.learning_rate = learning_rate
        self.weight_learning_rate = weight_learning_rate
        self.output_learning_rate = output_learning_rate

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def train_step(self, data_x, data_y):
        with tf.GradientTape() as tape:
            loss = tf.keras.losses.CategoricalCrossentropy()(data_y, self.model(data_x))

        gradients = tape.gradient(loss, self.model.trainable_variables)

        for i, grad in enumerate(gradients):
            if any(param in self.model.trainable_variables[i].name
                   for param in self.weight_params):
                gradients[i] = grad * self.weight_learning_rate / self.learning_rate
            elif any(param in self.model.trainable_variables[i].name
                     for param in self.output_params):
                gradients[i] = grad * self.output_learning_rate / self.learning_rate
            else:
                if self.no_grad and not isinstance(grad, tf.IndexedSlices):
                    gradients[i] = grad * 0.

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def model_weights_random_init(self, init_ratio=0.0001):
        weights = self.model.get_weights()

        for i, weight in enumerate(weights):
            if len(weight.shape) == 2:
                rand_mask = np.random.binomial(1, init_ratio, size=weight.shape)
                rand_weights = np.random.randn(*weight.shape) * rand_mask
                weights[i] = weight * (1 - rand_mask) + rand_weights

        self.model.set_weights(weights)

    def run_mono_train(self, epoch, per_batch, repeat=0):
        self.temp_val_acc = 0

        for (batch, (data_x, data_y)) in enumerate(self.train_dataset):
            self.train_step(data_x, data_y)

            if batch % per_batch == 0 and not batch == 0:
                prediction, label = [], []
                for x, y in tqdm(self.val_dataset):
                    prediction.append(self.model(x))
                    label.append(y)

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
seiya.set_dataset(batch_size=120*4)
# %%
num_layer_loops = 1
vector_dims = 128*1
num_heads = 4
inner_dims = 120
decoder_dims = vector_dims*2
seiya.set_model(num_layer_loops, vector_dims,
                num_heads, inner_dims, decoder_dims)
# %%
per_batch = 100

seiya.set_optimizer(learning_rate=2e-4,
                    weight_learning_rate=2e-2*0,
                    output_learning_rate=2e-4,
                    no_grad=False)

seiya.run_train(epochs=1, repeats=1, per_batch=per_batch)
# %%
seiya.best_val_loss = float('inf')
seiya.best_val_acc = 0

seiya.temp_val_loss = float('inf')
seiya.temp_val_acc = 0

seiya.set_optimizer(learning_rate=2e-4,
                    weight_learning_rate=2e-2,
                    output_learning_rate=2e-4,
                    no_grad=True)

seiya.run_train(epochs=1, repeats=1, per_batch=per_batch)
# %%
seiya.best_val_loss = float('inf')
seiya.best_val_acc = 0

seiya.temp_val_loss = float('inf')
seiya.temp_val_acc = 0

seiya.set_optimizer(learning_rate=2e-4,
                    weight_learning_rate=2e-2,
                    output_learning_rate=2e-4,
                    no_grad=False)

seiya.run_train(epochs=100, repeats=1, per_batch=per_batch)

# %%
layers.Softmax()(seiya.model.weight01)
# %%
layers.Softmax()(seiya.model.weight02)
# %%
