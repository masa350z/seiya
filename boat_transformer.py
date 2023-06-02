# %%
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


def split_data(inp, tr_rate=0.6, val_rate=0.2):
    train_len = int(len(inp)*tr_rate)
    valid_len = int(len(inp)*val_rate)

    train = inp[:train_len]
    valid = inp[train_len:train_len+valid_len]
    test = inp[train_len+valid_len:]

    return train, valid, test


def ret_known_racer(ar_num, unknown_rate=0.5, k_std=-2):
    known_racers, counts = np.unique(ar_num[:-int(len(ar_num)*unknown_rate)],
                                     return_counts=True)
    ave, std = np.average(counts), np.std(counts)
    known_racers = known_racers[counts > ave + std*k_std]

    known_bool = ar_num - known_racers.reshape(-1, 1, 1)
    known_bool = np.sum(known_bool == 0, axis=0)

    return ar_num*known_bool


def ret_norm00(ar):
    mx = np.max(ar)
    mn = np.min(ar)

    return (ar - mn)/(mx - mn)


class BoatNN(boatdata.BoatDataset):
    def __init__(self, unknown_rate=0.5, k_std=-2):
        super().__init__()

        self.racers = ret_known_racer(self.ar_num, unknown_rate, k_std)

        self.weather = self.ar_condition[:, 0].astype('int16')
        self.wind_vector = self.ar_condition[:, -1].astype('int16')

        conditions = self.ar_condition[:, 1:-1]

        mx = np.max(conditions, axis=0).reshape(1, -1)
        mn = np.min(conditions, axis=0).reshape(1, -1)

        self.conditions = (conditions - mn)/(mx - mn)

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

    def set_dataset(self, batch_size, mode='sanren', race_field=None):
        if race_field:
            field_filter = self.ar_field == race_field
        else:
            field_filter = np.tile(True, len(bt.ar_field))
        data_x = tf.data.Dataset.from_tensor_slices((bt.racers[field_filter], bt.ar_grade_num[field_filter],
                                                     bt.ar_ze[field_filter], bt.ar_to[field_filter],
                                                     bt.conditions[field_filter], bt.ar_incourse_num[field_filter],
                                                     bt.ar_field[field_filter]-1, bt.weather[field_filter]-1,
                                                     bt.wind_vector[field_filter],
                                                     bt.tenji_time[field_filter], bt.tenji_start_time[field_filter],
                                                     bt.sanren_odds[field_filter]))
        if mode == 'sanren':
            data_y = tf.data.Dataset.from_tensor_slices(bt.ret_sanren_onehot()[field_filter])
        else:
            data_y = tf.data.Dataset.from_tensor_slices(bt.ret_niren_onehot()[field_filter])
        dataset = tf.data.Dataset.zip((data_x, data_y))

        dataset_size = tf.data.experimental.cardinality(dataset).numpy()
        dataset = dataset.shuffle(dataset_size)

        train_size = int(0.6 * dataset_size)
        val_size = int(0.2 * dataset_size)

        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size).take(val_size)
        test_dataset = dataset.skip(train_size + val_size)

        self.train_dataset = train_dataset.batch(batch_size)
        self.val_dataset = val_dataset.batch(batch_size)
        self.test_dataset = test_dataset.batch(batch_size)


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

        self.racer_encoder01 = transformer.NoEmbeddingEncoder(num_layers=1, d_model=feature_dim*7,
                                                              num_heads=7, dff=feature_dim,
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

        self.odds_encoder01 = layers.Dense(36, activation='linear')
        self.odds_encoder02 = transformer.NoEmbeddingEncoder(num_layers=3, d_model=36,
                                                             num_heads=6, dff=feature_dim,
                                                             max_sequence_len=120)
        self.odds_encoder03 = layers.Dense(feature_dim, activation='relu')

        self.combo_layer = layers.Dense(120, activation='relu')

        self.output_layer01 = layers.Dense(120, activation='relu')
        self.output_layer02 = layers.Dense(120, activation='relu')
        self.output_layer03 = layers.Dense(120, activation='softmax')

    def call(self, input):
        racers, grades, r_ze, r_to, conditions, course, fields, weather, wind, tenji, start, sanren_odds = input

        racers = self.racers_embedding(racers)
        course = self.course_embedding(course)
        grades = self.grade_embedding(grades)
        r_ze = self.r_ze_embedding(r_ze)
        r_to = self.r_to_embedding(r_to)
        tenji = self.tenji_embedding(tf.expand_dims(tenji, 2))
        start = self.start_embedding(tf.expand_dims(start, 2))

        def ret_racer_vector(num):
            racer_vector = layers.concatenate([tf.expand_dims(racers[:, num], 1),
                                               tf.expand_dims(course[:, num], 1),
                                               tf.expand_dims(grades[:, num], 1),
                                               tf.expand_dims(r_ze[:, num], 1),
                                               tf.expand_dims(r_to[:, num], 1),
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

        out = self.output_layer01(tf.divide(combo, sanren_odds))
        out = self.output_layer02(out)

        return self.output_layer03(out)


class FirstOutputEncoder(tf.keras.Model):
    def __init__(self, units):
        super(FirstOutputEncoder, self).__init__()
        self.layer01 = layers.Dense(units, activation='relu')
        self.layer02 = layers.Dense(units, activation='relu')
        self.layer03 = layers.Dense(units, activation='relu')
        self.layer04 = layers.Dense(units, activation='relu')
        self.layer05 = layers.Dense(units, activation='relu')
        self.layer06 = layers.Dense(units, activation='relu')
        self.layer07 = layers.Dense(units, activation='relu')
        self.layer08 = layers.Dense(units, activation='relu')
        self.layer09 = layers.Dense(units, activation='relu')
        self.layer10 = layers.Dense(units, activation='relu')

    def call(self, input):
        x = self.layer01(input)
        x = self.layer02(x)
        x = self.layer03(x)
        #x = self.layer04(x)
        #x = self.layer05(x)
        #x = self.layer06(x)
        #x = self.layer07(x)
        #x = self.layer08(x)
        #x = self.layer09(x)
        #x = self.layer10(x)

        return x


class BetModel(tf.keras.Model):
    def __init__(self, feature_dim):
        super(BetModel, self).__init__()
        self.probability_encoder = FirstOutputEncoder(feature_dim)
        self.odds_encoder = FirstOutputEncoder(feature_dim)
        self.kitai_encoder = FirstOutputEncoder(feature_dim)

        self.output_layer01 = layers.Dense(feature_dim, activation='relu')
        #self.output_layer02 = layers.Dense(120, activation='relu')
        self.output_layer02 = layers.Dense(120, activation='sigmoid')

    def call(self, input):
        probability, odds, kitai = input
        probability = self.probability_encoder(probability)
        odds = self.odds_encoder(odds)
        kitai = self.kitai_encoder(kitai)

        x = probability + odds + kitai
        x = self.output_layer01(x)
        x = self.output_layer02(x)

        return x


# %%
race_field = None
bt = BoatNN(unknown_rate=0, k_std=1)
bt.set_dataset(batch_size=120, race_field=race_field)

EPOCHS = 100
k_freeze = 3
feature_dim = 128

weight_name = 'datas/pred_sanren/best_weights_f{}'.format(str(race_field).zfill(2))
# %%
best_val_loss = float('inf')

for i in range(10):
    temp_val_loss = float('inf')
    model = SEIYA(feature_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    if race_field:
        model.load_weights('datas/pred_sanren/best_weights')

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
            for (batch, (data_x, data_y)) in enumerate(bt.train_dataset.shuffle(1500)):
                loss = train_step(data_x, data_y)

                if batch % 400 == 0:
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
                    print(f"Repeat : {i+1}")
                    print(f"Temp valid loss : {temp_val_loss}")
                    print(f"Best valid loss : {best_val_loss}")

                    freeze = freeze - 1 if freeze > 0 else freeze

# %%
model.load_weights(weight_name)
# %%
train_loss, train_acc = model.evaluate(bt.train_dataset)
val_loss, val_acc = model.evaluate(bt.val_dataset)
test_loss, test_acc = model.evaluate(bt.test_dataset)

print(train_loss)
print(val_loss)
print(test_loss)

print(train_acc)
print(val_acc)
print(test_acc)
# %%
bt.sanren_odds
# %%
model = SEIYA(feature_dim)
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.load_weights(weight_name)
# %%
val_loss, val_acc = model.evaluate(bt.val_dataset)
val_acc
# %%
"""
res = []
for data in tqdm(dataset[:3]):
    for x, y in data:
        pred = model(x).numpy()
        for p in pred:
            res.append(p)

probability = np.array(res)

y_tr, y_vl, y_te = bt.data_y

res_tr = res[:len(y_tr)]
res_vl = res[len(y_tr):len(y_tr)+len(y_vl)]
res_te = res[len(y_tr)+len(y_vl):]
# %%
kitai = probability*odds_sanren
# %%
feature_dim = 120
bet_model = BetModel(feature_dim)
optimizer = tf.keras.optimizers.Adam()
bet_model.compile(optimizer=optimizer,
                  loss=custom_loss,
                  metrics=['accuracy'])
# %%
data_y = np.concatenate(bt.data_y)
data_y = data_y*odds_sanren
data_y = data_y.astype('float32')
# %%
index = np.arange(len(probability))
np.random.shuffle(index)
# %%
p_train, p_valid, p_test = split_data(probability[index])
o_train, o_valid, o_test = split_data(odds_sanren[index])
k_train, k_valid, k_test = split_data(kitai[index])
y_train, y_valid, y_test = split_data(data_y[index])
# %%
batch_size = 1200
train_x = tf.data.Dataset.from_tensor_slices((p_train,
                                              o_train,
                                              k_train)).batch(batch_size)
train_y = tf.data.Dataset.from_tensor_slices(-y_train+1).batch(batch_size)
train_dataset = tf.data.Dataset.zip((train_x, train_y))

valid_x = tf.data.Dataset.from_tensor_slices((p_valid,
                                              o_valid,
                                              k_valid)).batch(batch_size)
valid_y = tf.data.Dataset.from_tensor_slices(-y_valid+1).batch(batch_size)
valid_dataset = tf.data.Dataset.zip((valid_x, valid_y))

test_x = tf.data.Dataset.from_tensor_slices((p_test,
                                             o_test,
                                             k_test)).batch(batch_size)
test_y = tf.data.Dataset.from_tensor_slices(-y_test+1).batch(batch_size)
test_dataset = tf.data.Dataset.zip((test_x, test_y))

# %%
dataset = [train_dataset, valid_dataset, test_dataset, 'datas/pred_sanren/best_bet_weights']
EPOCHS = 100
best_val_loss = float('inf')
for i in range(1000):
    feature_dim = 120
    temp_val_loss = float('inf')
    bet_model = BetModel(feature_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=4e-5)
    bet_model.compile(optimizer=optimizer,
                      loss=custom_loss2)


    @tf.function
    def train_step(data_x, data_y):
        with tf.GradientTape() as tape:
            loss = custom_loss(data_y, bet_model(data_x))

        gradients = tape.gradient(loss, bet_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, bet_model.trainable_variables))

        return loss

    def model_weights_random_init(init_ratio=0.0001):
        weights = bet_model.get_weights()

        for i, weight in enumerate(weights):
            if len(weight.shape) == 2:
                rand_mask = np.random.binomial(1, init_ratio, size=weight.shape)
                rand_weights = np.random.randn(*weight.shape) * rand_mask
                weights[i] = weight * (1 - rand_mask) + rand_weights

        bet_model.set_weights(weights)


    freeze = k_freeze
    last_epoch = 0

    for epoch in range(EPOCHS):
        if epoch - last_epoch < 10:
            for (batch, (data_x, data_y)) in enumerate(dataset[0].shuffle(1500)):
                loss = train_step(data_x, data_y)
                if loss == 0:
                    break

                if batch % 100 == 0:
                    val_loss = bet_model.evaluate(dataset[1])

                    if val_loss < temp_val_loss:
                        last_epoch = epoch
                        freeze = 0
                        temp_val_loss = val_loss
                        temp_weights = bet_model.get_weights()

                        tr_loss = bet_model.evaluate(dataset[0])
                        if val_loss < best_val_loss and tr_loss < 0:
                            best_val_loss = val_loss
                            bet_model.save_weights(dataset[3])
                    else:
                        if freeze == 0:
                            bet_model.set_weights(temp_weights)
                            model_weights_random_init(init_ratio=0.0001)
                            freeze = k_freeze

                    print('')
                    print(dataset[3])
                    print(f"Repeat : {i+1}")
                    print(f"Temp valid loss : {temp_val_loss}")
                    print(f"Best valid loss : {best_val_loss}")

                    freeze = freeze - 1 if freeze > 0 else freeze
# %%
bet_model.load_weights(dataset[3])
tr_loss = bet_model.evaluate(dataset[0])
vl_loss = bet_model.evaluate(dataset[1])
te_loss = bet_model.evaluate(dataset[2])
print(tr_loss)
print(vl_loss)
print(te_loss)
# %%
pr = bet_model.predict(dataset[0])
# %%
pr.shape
# %%
np.sum(pr>0)
# %%
bet_model.compile(optimizer=optimizer,
                  loss=custom_loss2)
# %%
import pandas as pd

p = 0.05
o = 20

c1 = probability > p
c2 = odds_sanren > o

bet = probability*c1*c2

win = (data_y-1)*bet

np.sum(win)

asset = np.cumsum(np.sum(win, axis=1))


pd.DataFrame(asset).plot()

np.sum(win)/np.sum(bet)
# %%
np.sum(bet)/len(bet)
# %%
bt.ar_th
# %%
"""