# %%
import boatdata
import numpy as np
from keras import layers
from tqdm import tqdm
import tensorflow as tf


def split_data(inp, tr_rate=0.6, val_rate=0.2):
    train_len = int(len(inp)*tr_rate)
    valid_len = int(len(inp)*val_rate)

    train = inp[:train_len]
    valid = inp[train_len:train_len+valid_len]
    test = inp[train_len+valid_len:]

    return train, valid, test


class BoatNN(boatdata.BoatDataset):
    def __init__(self, unknown_rate=0.5, k_std=-2):
        super().__init__()

        known_racers, counts = np.unique(self.ar_num[:-int(len(self.ar_num)*unknown_rate)],
                                         return_counts=True)

        ave = np.average(counts)
        std = np.std(counts)

        known_racers = known_racers[counts > ave + std*k_std]

        known_bool = self.ar_num - known_racers.reshape(-1, 1, 1)
        known_bool = np.sum(known_bool == 0, axis=0)

        self.racers = self.ar_num*known_bool

        self.weather = self.ar_condition[:, 0].astype('int16')
        self.wind_vector = self.ar_condition[:, -1].astype('int16')

        conditions = self.ar_condition[:, 1:-1]

        mx = np.max(conditions, axis=0).reshape(1, -1)
        mn = np.min(conditions, axis=0).reshape(1, -1)

        self.conditions = (conditions - mn)/(mx - mn)

        def ret_norm00(ar):
            mx = np.max(ar)
            mn = np.min(ar)

            return (ar - mn)/(mx - mn)

        ar_ze1 = ret_norm00(self.ar_ze1).reshape(-1, 6, 1)
        ar_ze2 = ret_norm00(self.ar_ze2).reshape(-1, 6, 1)
        ar_ze3 = ret_norm00(self.ar_ze3).reshape(-1, 6, 1)

        ar_to1 = ret_norm00(self.ar_to1).reshape(-1, 6, 1)
        ar_to2 = ret_norm00(self.ar_to2).reshape(-1, 6, 1)
        ar_to3 = ret_norm00(self.ar_to3).reshape(-1, 6, 1)

        self.ar_ze = np.concatenate([ar_ze1, ar_ze2, ar_ze3], axis=2)
        self.ar_to = np.concatenate([ar_to1, ar_to2, ar_to3], axis=2)

        mx = np.max(self.tenji_time, axis=1).reshape(-1, 1)
        mn = np.min(self.tenji_time, axis=1).reshape(-1, 1)

        self.tenji_time = (self.tenji_time - mn)/(mx - mn)
        self.tenji_start_time = np.where(self.tenji_start_time == 1, 0, self.tenji_start_time)

        self.sanren_odds = np.array(self.sanren_odds, dtype='float32')

    def set_dataset(self, batch_size):
        tf_racers = split_data(self.racers)
        tf_grades = split_data(self.ar_grade_num)
        tf_shoritsu_ze = split_data(self.ar_ze)
        tf_shoritsu_to = split_data(self.ar_to)

        tf_fields = split_data(self.ar_field - 1)
        tf_weather = split_data(self.weather - 1)
        tf_wind = split_data(self.wind_vector)

        tf_conditions = split_data(self.conditions)
        tf_course = split_data(self.ar_incourse_num)
        tf_tenji = split_data(self.tenji_time)
        tf_start = split_data(self.tenji_start_time)

        tf_sanren_odds = split_data(self.sanren_odds)

        self.train = [tf_racers[0], tf_grades[0],
                      tf_shoritsu_ze[0], tf_shoritsu_to[0],
                      tf_conditions[0], tf_course[0],
                      tf_fields[0], tf_weather[0], tf_wind[0],
                      tf_tenji[0], tf_start[0],
                      tf_sanren_odds[0]]

        self.train_x = tf.data.Dataset.from_tensor_slices((tf_racers[0], tf_grades[0],
                                                           tf_shoritsu_ze[0], tf_shoritsu_to[0],
                                                           tf_conditions[0], tf_course[0],
                                                           tf_fields[0], tf_weather[0], tf_wind[0],
                                                           tf_tenji[0], tf_start[0],
                                                           tf_sanren_odds[0])).batch(batch_size)

        self.valid = [tf_racers[1], tf_grades[1],
                      tf_shoritsu_ze[1], tf_shoritsu_to[1],
                      tf_conditions[1], tf_course[1],
                      tf_fields[1], tf_weather[1], tf_wind[1],
                      tf_tenji[1], tf_start[1],
                      tf_sanren_odds[1]]

        self.valid_x = tf.data.Dataset.from_tensor_slices((tf_racers[1], tf_grades[1],
                                                           tf_shoritsu_ze[1], tf_shoritsu_to[1],
                                                           tf_conditions[1], tf_course[1],
                                                           tf_fields[1], tf_weather[1], tf_wind[1],
                                                           tf_tenji[1], tf_start[1],
                                                           tf_sanren_odds[1])).batch(batch_size)

        self.test = [tf_racers[2], tf_grades[2],
                     tf_shoritsu_ze[2], tf_shoritsu_to[2],
                     tf_conditions[2], tf_course[2],
                     tf_fields[2], tf_weather[2], tf_wind[2],
                     tf_tenji[2], tf_start[2],
                     tf_sanren_odds[2]]

        self.test_x = tf.data.Dataset.from_tensor_slices((tf_racers[2], tf_grades[2],
                                                          tf_shoritsu_ze[2], tf_shoritsu_to[2],
                                                          tf_conditions[2], tf_course[2],
                                                          tf_fields[2], tf_weather[2], tf_wind[2],
                                                          tf_tenji[2], tf_start[2],
                                                          tf_sanren_odds[2])).batch(batch_size)

        self.data_y = split_data(np.load('datas/sanren_label.npy'))

        self.train_y = tf.data.Dataset.from_tensor_slices(self.data_y[0]).batch(batch_size)
        self.valid_y = tf.data.Dataset.from_tensor_slices(self.data_y[1]).batch(batch_size)
        self.test_y = tf.data.Dataset.from_tensor_slices(self.data_y[2]).batch(batch_size)

        self.train_dataset = tf.data.Dataset.zip((self.train_x, self.train_y))
        self.valid_dataset = tf.data.Dataset.zip((self.valid_x, self.valid_y))
        self.test_dataset = tf.data.Dataset.zip((self.test_x, self.test_y))


class FullyConnected(tf.keras.Model):
    def __init__(self, units):
        super(FullyConnected, self).__init__()

        self.layer01 = layers.Dense(units, activation='relu')
        self.layer02 = layers.Dense(units, activation='relu')
        self.layer03 = layers.Dense(units, activation='relu')
        self.layer04 = layers.Dense(units, activation='relu')
        self.layer05 = layers.Dense(units, activation='relu')

    def call(self, input):
        x = self.layer01(input)
        x = self.layer02(x)
        x = self.layer03(x)
        x = self.layer04(x)
        x = self.layer05(x)

        return x


class SEIYA(tf.keras.Model):
    def __init__(self, feature_dim):
        super(SEIYA, self).__init__()

        self.racers_embedding = layers.Embedding(10000, feature_dim)
        self.course_embedding = layers.Embedding(6, feature_dim)
        self.grade_embedding = layers.Embedding(4, feature_dim)
        self.field_embedding = layers.Embedding(24, feature_dim)
        self.weather_embedding = layers.Embedding(6, feature_dim)
        self.wind_embedding = layers.Embedding(36, feature_dim)

        self.r_ze_embedding = layers.Dense(feature_dim, activation='relu')
        self.r_to_embedding = layers.Dense(feature_dim, activation='relu')

        self.tenji_embedding = layers.Dense(feature_dim, activation='relu')
        self.start_embedding = layers.Dense(feature_dim, activation='relu')

        self.grades_embedding = layers.Dense(feature_dim, activation='relu')
        self.condition_embedding = layers.Dense(feature_dim, activation='relu')

        self.racers_layer = FullyConnected(feature_dim)
        self.fields_layer = FullyConnected(feature_dim)
        self.odds_layer = FullyConnected(feature_dim)
        self.combo_layer = FullyConnected(feature_dim)

        self.output_layer01 = layers.Dense(120, activation='softmax')

    def call(self, input):
        racers, grades, r_ze, r_to, conditions, course, fields, weather, wind, tenji, start, sanren_odds = input

        racers = self.racers_embedding(racers)
        course = self.course_embedding(course)
        r_ze = self.r_ze_embedding(r_ze)
        r_to = self.r_to_embedding(r_to)
        tenji = self.tenji_embedding(tf.expand_dims(tenji, 2))
        start = self.start_embedding(tf.expand_dims(start, 2))
        #racers = layers.Flatten()(layers.concatenate([racers, course, r_ze, r_to, tenji, start], axis=2))
        #racers = layers.Flatten()(layers.concatenate([racers + course, r_ze, r_to, tenji, start], axis=2))
        racers = layers.Flatten()(racers + course + r_ze + r_to + tenji + start)
        racers = self.racers_layer(racers)

        fields = self.field_embedding(fields)
        weather = self.weather_embedding(weather)
        wind = self.wind_embedding(wind)
        grades = layers.Flatten()(self.grade_embedding(grades))
        
        grades = self.grades_embedding(grades)
        conditions = self.condition_embedding(conditions)

        #fields = layers.concatenate([fields, weather, wind, grades, conditions], axis=1)
        fields = fields + weather + wind + grades + conditions
        fields = self.fields_layer(fields)

        odds = self.odds_layer(sanren_odds)

        combo = layers.concatenate([racers, fields, odds], axis=1)
        combo = self.combo_layer(combo)

        out01 = self.output_layer01(combo)

        return out01


# %%
bt = BoatNN(k_std=-1)
bt.set_dataset(batch_size=120)

feature_dim = 1024
model = SEIYA(feature_dim)
optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# %%
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


EPOCHS = 100
k_freeze = 3

dataset = [bt.train_dataset, bt.valid_dataset, bt.test_dataset, 'datas/pred_sanren/best_weights']
# %%
best_val_loss = float('inf')

for i in range(50):
    temp_val_loss = float('inf')
    model = SEIYA(feature_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=4e-5)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    freeze = k_freeze
    last_epoch = 0

    for epoch in range(EPOCHS):
        if epoch - last_epoch < 5:
            model.fit(dataset[0].shuffle(buffer_size=1500).take(500))
            val_loss, val_acc = model.evaluate(dataset[1])

            if val_loss < temp_val_loss:
                last_epoch = epoch
                freeze = 0
                temp_val_loss = val_loss
                temp_weights = model.get_weights()

            else:
                if freeze == 0:
                    model.set_weights(temp_weights)
                    model_weights_random_init(init_ratio=0.0001)
                    freeze = k_freeze

            print(f"Repeat : {i+1}")
            print(f"Temp valid loss : {temp_val_loss}")
            print(f"Best valid loss : {best_val_loss}")
            freeze = freeze - 1 if freeze > 0 else freeze
        else:
            pass

        model.set_weights(temp_weights)
        model.save_weights(dataset[3])

    if temp_val_loss < best_val_loss:
        best_val_loss = temp_val_loss
        model.set_weights(temp_weights)
        best_weights = model.get_weights()
        model.save_weights(dataset[3])

model.set_weights(best_weights)
model.save_weights(dataset[3])

# %%
"""
# %%
model.load_weights(dataset[3])
res = []
for data in tqdm(dataset[:3]):
    for x, y in data:
        pred = model(x).numpy()
        for p in pred:
            res.append(p)
# %%
res = np.array(res)
# %%
y_tr, y_vl, y_te = bt.data_y
# %%
res_tr = res[:len(y_tr)]
res_vl = res[len(y_tr):len(y_tr)+len(y_vl)]
res_te = res[len(y_tr)+len(y_vl):]
# %%
mx = res_tr - np.max(res_tr, axis=1).reshape(-1, 1)
pred_tr = (mx == 0)*y_tr
# %%
np.sum(pred_tr)/len(pred_tr)
# %%
"""
