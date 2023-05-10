# %%
import boatdata
import numpy as np
from keras import layers
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

    def set_dataset(self, batch_size, buffer_size=150000):
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

        one_hot1 = tf.one_hot(self.ar_th-1, 6)[:, :1]
        one_hot2 = tf.one_hot(self.ar_th-1, 6)[:, 1:2]
        one_hot3 = tf.one_hot(self.ar_th-1, 6)[:, 2:3]
        self.data_y1 = split_data(tf.reshape(one_hot1, (-1, 6)))
        self.data_y2 = split_data(tf.reshape(one_hot2, (-1, 6)))
        self.data_y3 = split_data(tf.reshape(one_hot3, (-1, 6)))

        self.train_y1 = tf.data.Dataset.from_tensor_slices(self.data_y1[0]).batch(batch_size)
        self.valid_y1 = tf.data.Dataset.from_tensor_slices(self.data_y1[1]).batch(batch_size)
        self.test_y1 = tf.data.Dataset.from_tensor_slices(self.data_y1[2]).batch(batch_size)

        self.train_dataset1 = tf.data.Dataset.zip((self.train_x, self.train_y1)).shuffle(buffer_size)
        self.valid_dataset1 = tf.data.Dataset.zip((self.valid_x, self.valid_y1))
        self.test_dataset1 = tf.data.Dataset.zip((self.test_x, self.test_y1))

        self.train_y2 = tf.data.Dataset.from_tensor_slices(self.data_y2[0]).batch(batch_size)
        self.valid_y2 = tf.data.Dataset.from_tensor_slices(self.data_y2[1]).batch(batch_size)
        self.test_y2 = tf.data.Dataset.from_tensor_slices(self.data_y2[2]).batch(batch_size)

        self.train_dataset2 = tf.data.Dataset.zip((self.train_x, self.train_y2)).shuffle(buffer_size)
        self.valid_dataset2 = tf.data.Dataset.zip((self.valid_x, self.valid_y2))
        self.test_dataset2 = tf.data.Dataset.zip((self.test_x, self.test_y2))

        self.train_y3 = tf.data.Dataset.from_tensor_slices(self.data_y3[0]).batch(batch_size)
        self.valid_y3 = tf.data.Dataset.from_tensor_slices(self.data_y3[1]).batch(batch_size)
        self.test_y3 = tf.data.Dataset.from_tensor_slices(self.data_y3[2]).batch(batch_size)

        self.train_dataset3 = tf.data.Dataset.zip((self.train_x, self.train_y3)).shuffle(buffer_size)
        self.valid_dataset3 = tf.data.Dataset.zip((self.valid_x, self.valid_y3))
        self.test_dataset3 = tf.data.Dataset.zip((self.test_x, self.test_y3))


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

        self.racers_layer01 = layers.Dense(1024, activation='relu')
        self.racers_layer02 = layers.Dense(512, activation='relu')

        self.fields_layer01 = layers.Dense(1024, activation='relu')
        self.fields_layer02 = layers.Dense(512, activation='relu')

        self.odds_layer01 = layers.Dense(1024, activation='relu')
        self.odds_layer02 = layers.Dense(512, activation='relu')

        self.combo_layer01 = layers.Dense(1024, activation='relu')
        self.combo_layer02 = layers.Dense(512, activation='relu')

        self.output_layer01 = layers.Dense(6, activation='softmax')

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
        racers = self.racers_layer01(racers)
        racers = self.racers_layer02(racers)

        fields = self.field_embedding(fields)
        weather = self.weather_embedding(weather)
        wind = self.wind_embedding(wind)
        grades = layers.Flatten()(self.grade_embedding(grades))
        
        grades = self.grades_embedding(grades)
        conditions = self.condition_embedding(conditions)

        #fields = layers.concatenate([fields, weather, wind, grades, conditions], axis=1)
        fields = fields + weather + wind + grades + conditions
        fields = self.fields_layer01(fields)
        fields = self.fields_layer02(fields)

        odds = self.odds_layer01(sanren_odds)
        odds = self.odds_layer02(odds)

        combo = layers.concatenate([racers, fields, odds], axis=1)
        combo = self.combo_layer01(combo)
        combo = self.combo_layer02(combo)

        out01 = self.output_layer01(combo)

        return out01


# %%
bt = BoatNN(k_std=-1)
bt.set_dataset(batch_size=120)

feature_dim = 32
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

dataset1 = [bt.train_dataset1, bt.valid_dataset1, bt.test_dataset1, 'datas/pred_1th/best_weights']
dataset2 = [bt.train_dataset2, bt.valid_dataset2, bt.test_dataset2, 'datas/pred_2th/best_weights']
dataset3 = [bt.train_dataset3, bt.valid_dataset3, bt.test_dataset3, 'datas/pred_3th/best_weights']

dataset = dataset1

freeze = k_freeze
best_val_loss = float('inf')
last_epoch = 0

for epoch in range(EPOCHS):
    if epoch - last_epoch < 10:
        for (batch, (data_x, data_y)) in enumerate(dataset[0]):
            loss = train_step(data_x, data_y)

            if batch % 300 == 0:
                print(f'Epoch {epoch+1} Batch {batch} Loss {loss.numpy():.4f}')

                val_loss, val_acc = model.evaluate(dataset[1])

                if val_loss < best_val_loss:
                    last_epoch = epoch
                    freeze = 0
                    best_val_loss = val_loss
                    best_weights = model.get_weights()
                    print(f"Valid loss decreased to {val_loss}, saving weights.")

                else:
                    if freeze == 0:
                        model.set_weights(best_weights)
                        model_weights_random_init(init_ratio=0.001)
                        freeze = k_freeze
                        print("Valid loss did not decrease, loading weights.")
                    else:
                        print("Valid loss did not decrease.")

                freeze = freeze - 1 if freeze > 0 else freeze
    else:
        break

    model.set_weights(best_weights)
    model.save_weights(dataset[3])

# %%
model.set_weights(best_weights)
model.evaluate(dataset[1])
# %%
model.evaluate(dataset[2])
# %%
