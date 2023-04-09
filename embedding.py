# %%
from transformers import TFBertModel, TFAlbertModel, TFGPT2Model
from keras.layers import Dense, Embedding, concatenate
from keras.models import Model
import tensorflow as tf

from tqdm import tqdm
import numpy as np
import boatdata


class Sanren120(boatdata.BoatDataset):
    """
    ラベルが3連単の組み合わせ120通りのデータセット
    (x, 120)次元のラベル
    """
    def __init__(self, ret_grade=True, sorted=True):
        super().__init__(ret_grade, sorted)
        self.set_label()

    def set_label(self):
        label = np.zeros((len(self.ar_num), 120))
        th_ar = self.ret_sorted_th()[:, :3]

        for i in tqdm(range(len(th_ar))):
            temp_ar = th_ar[i]

            sanren = ''
            for nm in temp_ar:
                sanren += str(nm)

            label[i][self.sanren_dic[sanren]] = 1

        self.label = label

    def set_dataset(self, batch_size):
        mask = np.array([False, False, False, False, False, False, False, False,
                         False, True, True, True, True, True, True, True, True, True, True, True, True, False])
        inp = self.tokenized_inputs[:,mask]

        mn = np.min(np.where(inp == 100, 100000, inp))

        inp = inp - mn + 1
        inp = np.where(inp == -11500, 0, inp)

        pre_mask = np.array([True, True, True,  False, False, False,  True,  True])
        pre_info = self.pre_info[:,:,pre_mask]

        self.x_train, self.x_valid, self.x_test = boatdata.split_data(inp)
        self.pre_train, self.pre_valid, self.pre_test = boatdata.split_data(pre_info)
        self.y_train, self.y_valid, self.y_test = boatdata.split_data(self.label)
        """
        self.train = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).batch(batch_size)
        self.valid = tf.data.Dataset.from_tensor_slices((self.x_valid, self.y_valid)).batch(batch_size)
        self.test = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(batch_size)
        """
        x = tf.data.Dataset.from_tensor_slices((self.x_train)).batch(batch_size)
        pre = tf.data.Dataset.from_tensor_slices((self.pre_train)).batch(batch_size)
        train_y = tf.data.Dataset.from_tensor_slices((self.y_train)).batch(batch_size)
        self.train = tf.data.Dataset.zip((x, pre))
        self.train = tf.data.Dataset.zip((self.train, train_y))

        x = tf.data.Dataset.from_tensor_slices((self.x_valid)).batch(batch_size)
        pre = tf.data.Dataset.from_tensor_slices((self.pre_valid)).batch(batch_size)
        valid_y = tf.data.Dataset.from_tensor_slices((self.y_valid)).batch(batch_size)
        self.valid = tf.data.Dataset.zip((x, pre))
        self.valid = tf.data.Dataset.zip((self.valid, valid_y))
        
        x = tf.data.Dataset.from_tensor_slices((self.x_test)).batch(batch_size)
        pre = tf.data.Dataset.from_tensor_slices((self.pre_test)).batch(batch_size)
        test_y = tf.data.Dataset.from_tensor_slices((self.y_test)).batch(batch_size)
        self.test = tf.data.Dataset.zip((x, pre))
        self.test = tf.data.Dataset.zip((self.test, test_y))


    def model_compile(self, learning_rate=False):
        if learning_rate:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = 'Adam'

        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def start_training(self, epochs, weight_name, k_freeze=3):
        best_val_loss = float('inf')
        k_freeze = 3
        freeze = k_freeze

        val_loss, val_acc = self.model.evaluate(self.valid)
        print(f"Initial valid loss: {val_loss}")

        # 学習を開始する
        for epoch in range(epochs):
            self.model.fit(self.train)
            val_loss, val_acc = self.model.evaluate(self.valid)

            # valid lossが減少した場合、重みを保存
            if val_loss < best_val_loss:
                freeze = 0
                best_val_loss = val_loss
                self.model.save_weights(weight_name)
                print(f"Epoch {epoch + 1}: Valid loss decreased to {val_loss}, saving weights.")

            # valid lossが減少しなかった場合、保存しておいた最良の重みをロード
            else:
                if freeze == 0:
                    self.model.load_weights(weight_name)
                    self.model_weights_random_init()
                    freeze = k_freeze
                    print(f"Epoch {epoch + 1}: Valid loss did not decrease, loading weights.")
                else:
                    print(f"Epoch {epoch + 1}: Valid loss did not decrease.")

            freeze = freeze - 1 if freeze > 0 else freeze

            print('')

# %%
bt= Sanren120()
# %%


class Boat_NLP(Model):
    def __init__(self):
        super(Boat_NLP, self).__init__(name='boat_nlp')
        self.vect_len = 2048

        self.embedding = Embedding(1382, self.vect_len)

        self.senshu01 = Dense(1024, activation='relu')
        self.senshu02 = Dense(1024, activation='relu')
        self.senshu03 = Dense(1024, activation='relu')
        self.senshu04 = Dense(1024, activation='relu')
        self.senshu05 = Dense(1024, activation='relu')
        self.senshu06 = Dense(1024, activation='relu')

        self.layer01 = Dense(1024*5, activation='relu')
        self.layer02 = Dense(512*5, activation='relu')
        self.layer03 = Dense(256*5, activation='relu')
        self.layer04 = Dense(128*5, activation='relu')

        self.pre01 = Dense(1024, activation='relu')
        self.pre02 = Dense(1024, activation='relu')
        self.pre03 = Dense(1024, activation='relu')
        self.pre04 = Dense(1024, activation='relu')
        self.pre05 = Dense(1024, activation='relu')
        self.pre06 = Dense(1024, activation='relu')

        self.output_layer = Dense(120, activation='softmax')

    def call(self, inputs):
        x, pre = inputs
        x = self.embedding(x)

        pre01 = self.pre01(pre[:, 0])
        pre02 = self.pre02(pre[:, 1])
        pre03 = self.pre03(pre[:, 2])
        pre04 = self.pre04(pre[:, 3])
        pre05 = self.pre05(pre[:, 4])
        pre06 = self.pre06(pre[:, 5])

        x01 = self.senshu01(concatenate([tf.reshape(x[:, 0:2], (-1, 2*self.vect_len)), pre01]))
        x02 = self.senshu02(concatenate([tf.reshape(x[:, 2:4], (-1, 2*self.vect_len)), pre02]))
        x03 = self.senshu03(concatenate([tf.reshape(x[:, 4:6], (-1, 2*self.vect_len)), pre03]))
        x04 = self.senshu04(concatenate([tf.reshape(x[:, 6:8], (-1, 2*self.vect_len)), pre04]))
        x05 = self.senshu05(concatenate([tf.reshape(x[:, 8:10], (-1, 2*self.vect_len)), pre05]))
        x06 = self.senshu06(concatenate([tf.reshape(x[:, 10:12], (-1, 2*self.vect_len)), pre06]))

        x = self.layer01(concatenate([x01, x02, x03, x04, x05, x06]))
        x = self.layer02(x)
        x = self.layer03(x)
        x = self.layer04(x)
        x = self.output_layer(x)

        return x

# %%
bt.model = Boat_NLP()
bt.set_dataset(batch_size=12)
bt.model_compile(learning_rate=2e-5)
# %%
bt.start_training(epochs=100, weight_name='datas/sanren120/boat_nlp')
# %%
bt.pre_info[0][0]
# %%

# %%
