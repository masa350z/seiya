# %%
import os
import boatdata
import numpy as np
from keras import layers
import tensorflow as tf
import transformer


def custom_loss(y_true, y_pred):
    loss = tf.multiply(y_true, y_pred)
    lose = tf.cast(loss[loss > 0], dtype=tf.float32)
    win = tf.cast(loss[loss < 0], dtype=tf.float32)

    return (tf.reduce_sum(tf.math.sqrt(lose)) - tf.reduce_sum(tf.math.sqrt(-win)))


def custom_loss2(y_true, y_pred):
    loss = tf.multiply(y_true, y_pred)
    return tf.reduce_mean(tf.reduce_sum(loss, axis=1))


class SeiyaEncoder(tf.keras.Model):
    def __init__(self, feature_dim):
        super(SeiyaEncoder, self).__init__()

        self.racers_embedding = layers.Embedding(10000, feature_dim)
        self.course_embedding = layers.Embedding(6, feature_dim)
        self.grade_embedding = layers.Embedding(4, feature_dim)
        self.field_embedding = layers.Embedding(24, feature_dim)
        self.weather_embedding = layers.Embedding(6, feature_dim)
        self.wind_embedding = layers.Embedding(36, feature_dim)

        self.r_ze01_embedding = layers.Dense(feature_dim, activation='relu')
        self.r_ze02_embedding = layers.Dense(feature_dim, activation='relu')
        self.r_ze03_embedding = layers.Dense(feature_dim, activation='relu')
        self.r_ze_embedding = layers.Dense(feature_dim, activation='relu')

        self.r_to01_embedding = layers.Dense(feature_dim, activation='relu')
        self.r_to02_embedding = layers.Dense(feature_dim, activation='relu')
        self.r_to03_embedding = layers.Dense(feature_dim, activation='relu')
        self.r_to_embedding = layers.Dense(feature_dim, activation='relu')

        self.tenji_embedding = layers.Dense(feature_dim, activation='relu')
        self.start_embedding = layers.Dense(feature_dim, activation='relu')

        self.racer_encoder01 = transformer.NoEmbeddingEncoder(num_layers=1, d_model=feature_dim*11,
                                                              num_heads=11, dff=feature_dim,
                                                              max_sequence_len=6)

        self.encoder01 = layers.Dense(feature_dim, activation='relu')
        self.encoder02 = layers.Dense(feature_dim, activation='relu')
        self.encoder03 = layers.Dense(feature_dim, activation='relu')
        self.encoder04 = layers.Dense(feature_dim, activation='relu')
        self.encoder05 = layers.Dense(feature_dim, activation='relu')
        self.encoder06 = layers.Dense(feature_dim, activation='relu')

        self.racer_encoder02 = layers.Dense(feature_dim, activation='relu')
        self.field_encoder = layers.Dense(feature_dim, activation='relu')
        self.odds_encoder = layers.Dense(feature_dim, activation='linear')

        self.condition_embedding = layers.Dense(feature_dim, activation='relu')

        self.odds_encoder01 = layers.Dense(12, activation='linear')
        self.odds_encoder02 = transformer.NoEmbeddingEncoder(num_layers=3, d_model=12,
                                                             num_heads=6, dff=feature_dim,
                                                             max_sequence_len=120)
        self.odds_encoder03 = layers.Dense(feature_dim, activation='relu')

        self.combo_layer = layers.Dense(120, activation='relu')

    def call(self, input):
        racers, grades, r_ze, r_to, conditions, course, fields, weather, wind, tenji, start, sanren_odds = input

        racers = self.racers_embedding(racers)
        course = self.course_embedding(course)
        grades = self.grade_embedding(grades)

        r_ze01 = self.r_ze01_embedding(tf.expand_dims(r_ze[:, 0], 2))
        r_ze02 = self.r_ze02_embedding(tf.expand_dims(r_ze[:, 1], 2))
        r_ze03 = self.r_ze03_embedding(tf.expand_dims(r_ze[:, 2], 2))

        r_to01 = self.r_to01_embedding(tf.expand_dims(r_to[:, 0], 2))
        r_to02 = self.r_to02_embedding(tf.expand_dims(r_to[:, 1], 2))
        r_to03 = self.r_to03_embedding(tf.expand_dims(r_to[:, 2], 2))

        tenji = self.tenji_embedding(tf.expand_dims(tenji, 2))
        start = self.start_embedding(tf.expand_dims(start, 2))

        def ret_racer_vector(num):
            racer_vector = layers.concatenate([tf.expand_dims(racers[:, num], 1),
                                               tf.expand_dims(course[:, num], 1),
                                               tf.expand_dims(grades[:, num], 1),
                                               tf.expand_dims(r_ze01[:, num], 1),
                                               tf.expand_dims(r_ze02[:, num], 1),
                                               tf.expand_dims(r_ze03[:, num], 1),
                                               tf.expand_dims(r_to01[:, num], 1),
                                               tf.expand_dims(r_to02[:, num], 1),
                                               tf.expand_dims(r_to03[:, num], 1),
                                               tf.expand_dims(tenji[:, num], 1),
                                               tf.expand_dims(start[:, num], 1),
                                               ], axis=1)
            return racer_vector

        racer_vector01 = layers.Flatten()(ret_racer_vector(0))
        racer_vector02 = layers.Flatten()(ret_racer_vector(1))
        racer_vector03 = layers.Flatten()(ret_racer_vector(2))
        racer_vector04 = layers.Flatten()(ret_racer_vector(3))
        racer_vector05 = layers.Flatten()(ret_racer_vector(4))
        racer_vector06 = layers.Flatten()(ret_racer_vector(5))

        racer_vectors = layers.concatenate([tf.expand_dims(racer_vector01, 1),
                                            tf.expand_dims(racer_vector02, 1),
                                            tf.expand_dims(racer_vector03, 1),
                                            tf.expand_dims(racer_vector04, 1),
                                            tf.expand_dims(racer_vector05, 1),
                                            tf.expand_dims(racer_vector06, 1),
                                            ], axis=1)

        racers = self.racer_encoder01(racer_vectors)

        racer_vector01 = self.encoder01(racers[:, 0])
        racer_vector02 = self.encoder02(racers[:, 1])
        racer_vector03 = self.encoder03(racers[:, 2])
        racer_vector04 = self.encoder04(racers[:, 3])
        racer_vector05 = self.encoder05(racers[:, 4])
        racer_vector06 = self.encoder06(racers[:, 5])

        racers = self.racer_encoder02(layers.concatenate([racer_vector01,
                                                          racer_vector02,
                                                          racer_vector03,
                                                          racer_vector04,
                                                          racer_vector05,
                                                          racer_vector06,
                                                          ]))

        fields = self.field_embedding(fields)
        weather = self.weather_embedding(weather)
        wind = self.wind_embedding(wind)
        conditions = self.condition_embedding(conditions)

        fields = layers.concatenate([fields, weather, wind, conditions])
        fields = self.field_encoder(fields)

        odds = self.odds_encoder01(tf.expand_dims(sanren_odds, 2))
        odds = self.odds_encoder02(odds)
        odds = self.odds_encoder03(layers.Flatten()(odds))

        combo = layers.concatenate([racers, fields, odds])
        combo = self.combo_layer(combo)

        return combo, sanren_odds


class SeiyaOnehot(tf.keras.Model):
    def __init__(self, feature_dim, output_dim):
        super(SeiyaOnehot, self).__init__()
        
        self.encoder = SeiyaEncoder(feature_dim)

        self.output_layer01 = layers.Dense(120, activation='relu')
        self.output_layer02 = layers.Dense(120, activation='relu')
        self.output_layer03 = layers.Dense(output_dim, activation='softmax')
    
    def call(self, input):
        x, sanren_odds = self.encoder(input)
        x = self.output_layer01(tf.divide(x, sanren_odds))
        x = self.output_layer02(x)

        return self.output_layer03(x)


class SeiyaBet(tf.keras.Model):
    def __init__(self, feature_dim, output_dim, output_activation='softmax'):
        super(SeiyaBet, self).__init__()
        
        self.encoder = SeiyaEncoder(feature_dim)

        self.output_layer01 = layers.Dense(120, activation='relu')
        self.output_layer02 = layers.Dense(120, activation='relu')
        self.output_layer03 = layers.Dense(output_dim, activation=output_activation)
    
    def call(self, input):
        x, sanren_odds = self.encoder(input)
        x = self.output_layer01(tf.multiply(x, sanren_odds))
        x = self.output_layer02(x)

        return self.output_layer03(x)


# %%
class NNDataSet(boatdata.BoatDataset):
    def __init__(self, unknown_rate=0., k_std=-10**10, race_field=None):
        super().__init__(race_field)

        self.racers = self.ret_known_racer(unknown_rate, k_std)

        self.weather = self.ar_condition[:, 0].astype('int16')
        self.wind_vector = self.ar_condition[:, -1].astype('int16')

        conditions = self.ar_condition[:, 1:-1]

        mx = np.max(conditions, axis=0).reshape(1, -1)
        mn = np.min(conditions, axis=0).reshape(1, -1)

        self.conditions = (conditions - mn)/(mx - mn)

        ar_ze1 = np.expand_dims(self.ar_ze1, 1)
        ar_ze2 = np.expand_dims(self.ar_ze2, 1)
        ar_ze3 = np.expand_dims(self.ar_ze3, 1)

        ar_to1 = np.expand_dims(self.ar_to1, 1)
        ar_to2 = np.expand_dims(self.ar_to2, 1)
        ar_to3 = np.expand_dims(self.ar_to3, 1)

        self.ar_ze = np.concatenate([ar_ze1, ar_ze2, ar_ze3], axis=1)
        self.ar_to = np.concatenate([ar_to1, ar_to2, ar_to3], axis=1)

        mx = np.max(self.tenji_time, axis=1).reshape(-1, 1)
        mn = np.min(self.tenji_time, axis=1).reshape(-1, 1)

        self.tenji_time = (self.tenji_time - mn)/(mx - mn)
        self.tenji_start_time = np.where(self.tenji_start_time == 1, 0, self.tenji_start_time)

    def ret_known_racer(self, unknown_rate=0.5, k_std=-2):
        known_racers, counts = np.unique(self.ar_num[:-int(len(self.ar_num)*unknown_rate)],
                                        return_counts=True)
        ave, std = np.average(counts), np.std(counts)
        known_racers = known_racers[counts > ave + std*k_std]

        known_bool = self.ar_num - known_racers.reshape(-1, 1, 1)
        known_bool = np.sum(known_bool == 0, axis=0)

        return self.ar_num*known_bool

    def ret_datax(self):
        data_x = tf.data.Dataset.from_tensor_slices((self.racers, self.ar_grade_num,
                                                     self.ar_ze, self.ar_to,
                                                     self.conditions, self.ar_incourse_num,
                                                     self.ar_field-1, self.weather-1,
                                                     self.wind_vector,
                                                     self.tenji_time, self.tenji_start_time,
                                                     self.sanren_odds))
        
        return data_x

    def set_dataset(self, batch_size, train_rate=0.6, val_rate=0.2, shuffle=False):
        dataset = tf.data.Dataset.zip((self.data_x, self.data_y))

        dataset_size = tf.data.experimental.cardinality(dataset).numpy()
        if shuffle:
            dataset = dataset.shuffle(dataset_size)

        self.train_size = int(train_rate * dataset_size)
        self.val_size = int(val_rate * dataset_size)

        train_dataset = dataset.take(self.train_size)
        val_dataset = dataset.skip(self.train_size).take(self.val_size)
        test_dataset = dataset.skip(self.train_size + self.val_size)

        self.train_dataset = train_dataset.batch(batch_size)
        self.val_dataset = val_dataset.batch(batch_size)
        self.test_dataset = test_dataset.batch(batch_size)


class SanrenDataset(NNDataSet):
    def __init__(self, unknown_rate=0.5, k_std=-2, race_field=None, sanren_index_array=None):
        super().__init__(unknown_rate, k_std, race_field)
        
        if not sanren_index_array:
            sorted123 = self.ret_sorted123()
            self.sanren_index_array = np.array([np.where(sorted123 == i)[0][0] for i in self.sanren_indx])
        else:
            self.sanren_index_array = sanren_index_array
        
        self.data_x = self.ret_datax()

    def ret_sorted123(self):
        ar123 = self.ar_th[:, :3]
        ar123 = ar123[:, 0]*100 + ar123[:, 1]*10 + ar123[:, 2]*1
        ar123, count123 = np.unique(ar123, return_counts=True)

        sorted123 = ar123[np.argsort(count123)[::-1]]
        
        return sorted123
    
    def set_datay_onehot(self, output_size=None):
        data_y = self.ret_sanren_onehot()
        
        if output_size:
            data_y = data_y[:, self.sanren_index_array][:, :output_size]
            
            other_y = np.sum(data_y, axis=1)==0
            data_y = np.concatenate([data_y, other_y.reshape(-1, 1)*1], axis=1)
    
        self.data_y = tf.data.Dataset.from_tensor_slices(data_y)

    def set_datay_odds(self, output_size=None):
        data_y_odds = self.ret_all_sanren_odds()
        
        if output_size:
            data_y_odds = data_y_odds[:, self.sanren_index_array][:, :output_size]
            data_y_odds = np.concatenate([data_y_odds, np.ones((len(data_y), 1), dtype='float32')], axis=1)
    
        self.data_y = tf.data.Dataset.from_tensor_slices(data_y_odds*data_y)


class NirenDataset(NNDataSet):
    def __init__(self, unknown_rate=0.5, k_std=-2, race_field=None, niren_index_array=None):
        super().__init__(unknown_rate, k_std, race_field)
        
        if not niren_index_array:
            sorted12 = self.ret_sorted12()
            self.niren_index_array = np.array([np.where(sorted12 == i)[0][0] for i in self.niren_indx])
        else:
            self.niren_index_array = niren_index_array

    def ret_sorted12(self):
        ar12 = self.ar_th[:, :2]
        ar12 = ar12[:, 0]*10 + ar12[:, 1]*1
        ar12, count12 = np.unique(ar12, return_counts=True)

        sorted12 = ar12[np.argsort(count12)[::-1]]
        
        return sorted12

# %%
bt = SanrenDataset(race_field=None)
# %%
bt.set_datay_onehot(output_size=None)
bt.set_dataset(batch_size=120, shuffle=True)

EPOCHS = 100
k_freeze = 3
feature_dim = 120

weight_name = 'datas/pred_sanren/all/best_weights'
os.makedirs('datas/pred_sanren/all', exist_ok=True)

best_val_loss = float('inf')

for i in range(1000):
    temp_val_loss = float('inf')
    model = SeiyaOnehot(feature_dim, 120)
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
    model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    @tf.function
    def train_step(data_x, data_y):
        with tf.GradientTape() as tape:
            loss = tf.keras.losses.CategoricalCrossentropy()(data_y, model(data_x))

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss

    def model_weights_random_init(init_ratio=0.0001):
        weights = model.get_weights()

        for i, weight in enumerate(weights):
            if len(weight.shape) == 2:
                rand_mask = np.random.binomial(1, init_ratio, size=weight.shape)
                rand_weights = np.random.randn(*weight.shape) * rand_mask
                weights[i] = weight * (1 - rand_mask) + rand_weights

        model.set_weights(weights)

    freeze = k_freeze
    last_epoch = 0

    for epoch in range(EPOCHS):
        if epoch - last_epoch < 5:
            for (batch, (data_x, data_y)) in enumerate(bt.train_dataset.shuffle(bt.train_size)):
                loss = train_step(data_x, data_y)

                if batch % 600 == 0:
                    val_loss, val_acc = model.evaluate(bt.val_dataset)

                    if val_loss < temp_val_loss:
                        last_epoch = epoch
                        freeze = 0
                        temp_val_loss = val_loss
                        temp_weights = model.get_weights()

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            model.save_weights(weight_name)
                    else:
                        if freeze == 0:
                            model.set_weights(temp_weights)
                            model_weights_random_init(init_ratio=0.0001)
                            freeze = k_freeze

                    print('')
                    print(weight_name)
                    print(f"Repeat : {i + 1}")
                    print(f"Temp valid loss : {temp_val_loss}")
                    print(f"Best valid loss : {best_val_loss}")

                    freeze = freeze - 1 if freeze > 0 else freeze

# %%
model.load_weights(weight_name)
# %%
bt.set_dataset(batch_size=120, shuffle=False)
res = model.predict(bt.val_dataset)

data_y = []
for x, y in bt.val_dataset:
    data_y.append(y.numpy())

data_y = np.concatenate(data_y)
bet = (res - np.max(res, axis=1).reshape(-1, 1))== 0

np.sum(data_y*bet)/len(bet)

# %%
res1 = res[:, :-1]
res2 = res[:, -1]
# %%
bet1 = (res1 - np.max(res1, axis=1).reshape(-1, 1))== 0
# %%
result = bet1*data_y[:, :-1]
# %%
k = 0.4
a = result[res2 < k]
np.sum(a)/len(a)
# %%
len(a)/len(res)
# %%
