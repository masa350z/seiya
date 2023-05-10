# %%
from transformers import TFBertModel
from keras.layers import Dense, concatenate
from keras.models import Model
import tensorflow as tf

import numpy as np
import boatdata


class Sanren6_6(boatdata.BoatDataset):
    def __init__(self, ret_grade=True, sorted=True):
        super().__init__(ret_grade, sorted)
        self.set_label()

    def set_label(self):
        sorted_th = self.ret_sorted_th()

        data_y = []
        arr = np.array([0, 1, 2, 3, 4, 5])
        for i in range(len(sorted_th)):
            temp = sorted_th[i]

            co_to_th = [arr[temp == j+1][0] for j in range(6)]

            y_ = np.zeros((6, 6))
            for j in range(6):
                y_[j][co_to_th[j]] = 1

            data_y.append(y_)

        self.label = np.array(data_y)

    def set_dataset(self, batch_size):
        self.x_train, self.x_valid, self.x_test = boatdata.split_data(self.tokenized_inputs)
        self.pre_train, self.pre_valid, self.pre_test = boatdata.split_data(self.pre_info)
        self.y_train, self.y_valid, self.y_test = boatdata.split_data(self.label)

        x_train_ = tf.data.Dataset.from_tensor_slices(self.x_train).batch(batch_size)
        x_valid_ = tf.data.Dataset.from_tensor_slices(self.x_valid).batch(batch_size)
        x_test_ = tf.data.Dataset.from_tensor_slices(self.x_test).batch(batch_size)

        pre_train_ = tf.data.Dataset.from_tensor_slices(self.pre_train).batch(batch_size)
        pre_valid_ = tf.data.Dataset.from_tensor_slices(self.pre_valid).batch(batch_size)
        pre_test_ = tf.data.Dataset.from_tensor_slices(self.pre_test).batch(batch_size)

        y_train_ = tf.data.Dataset.from_tensor_slices(self.y_train).batch(batch_size)
        y_valid_ = tf.data.Dataset.from_tensor_slices(self.y_valid).batch(batch_size)
        y_test_ = tf.data.Dataset.from_tensor_slices(self.y_test).batch(batch_size)

        self.train = tf.data.Dataset.zip((x_train_, pre_train_))
        self.valid = tf.data.Dataset.zip((x_valid_, pre_valid_))
        self.test = tf.data.Dataset.zip((x_test_, pre_test_))

        self.train = tf.data.Dataset.zip((self.train, y_train_))
        self.valid = tf.data.Dataset.zip((self.valid, y_valid_))
        self.test = tf.data.Dataset.zip((self.test, y_test_))

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

            pred = self.model.predict(self.valid)
            y = self.y_valid

            mx = np.expand_dims(np.max(pred, axis=2), 2)

            win = ((pred - mx) == 0)*y
            win = np.sum(win, axis=2)

            print(np.sum(win[:, 0])/len(win))
            print(np.sum(win[:, 1])/len(win))
            print(np.sum(win[:, 2])/len(win))
            print(np.sum(win[:, 3])/len(win))
            print(np.sum(win[:, 4])/len(win))
            print(np.sum(win[:, 5])/len(win))

            print('')


class RNN_Boat_Bert(Model):
    def __init__(self, bert_model='bert-base-uncased'):
        super(RNN_Boat_Bert, self).__init__(name='boat_bert')

        self.bert_model = TFBertModel.from_pretrained(bert_model)

        self.dense01 = Dense(512, activation='relu')
        self.dense02 = Dense(512, activation='relu')
        self.dense03 = Dense(512, activation='relu')
        self.dense04 = Dense(512, activation='relu')
        self.dense05 = Dense(512, activation='relu')
        self.dense06 = Dense(512, activation='relu')

        self.dense01_ = Dense(128, activation='relu')
        self.dense02_ = Dense(128, activation='relu')
        self.dense03_ = Dense(128, activation='relu')
        self.dense04_ = Dense(128, activation='relu')
        self.dense05_ = Dense(128, activation='relu')
        self.dense06_ = Dense(128, activation='relu')

        self.conc01 = Dense(256, activation='relu')
        self.conc02 = Dense(256, activation='relu')
        self.conc03 = Dense(256, activation='relu')
        self.conc04 = Dense(256, activation='relu')
        self.conc05 = Dense(256, activation='relu')
        self.conc06 = Dense(256, activation='relu')

        self.output_01 = Dense(6, activation='softmax')
        self.output_02 = Dense(6, activation='softmax')
        self.output_03 = Dense(6, activation='softmax')
        self.output_04 = Dense(6, activation='softmax')
        self.output_05 = Dense(6, activation='softmax')
        self.output_06 = Dense(6, activation='softmax')

    def call(self, inputs):
        x, pre = inputs
        x = self.bert_model(x)

        x1 = self.dense01(x[0][:, 9])
        pre1 = self.dense01_(pre[:, 0])
        x1_ = self.conc01(concatenate([x1, pre1]))
        x1 = self.output_01(x1_)

        x2 = self.dense02(x[0][:, 11])
        pre2 = self.dense02_(pre[:, 1])
        x2_ = self.conc02(concatenate([x2, pre2, x1_]))
        x2 = self.output_02(x2_)

        x3 = self.dense03(x[0][:, 13])
        pre3 = self.dense03_(pre[:, 2])
        x3_ = self.conc03(concatenate([x3, pre3, x2_]))
        x3 = self.output_03(x3_)

        x4 = self.dense04(x[0][:, 15])
        pre4 = self.dense04_(pre[:, 3])
        x4_ = self.conc04(concatenate([x4, pre4, x3_]))
        x4 = self.output_04(x4_)

        x5 = self.dense05(x[0][:, 17])
        pre5 = self.dense05_(pre[:, 4])
        x5_ = self.conc05(concatenate([x5, pre5, x4_]))
        x5 = self.output_05(x5_)

        x6 = self.dense06(x[0][:, 19])
        pre6 = self.dense06_(pre[:, 5])
        x6_ = self.conc06(concatenate([x6, pre6, x5_]))
        x6 = self.output_06(x6_)

        return tf.stack([x1, x2, x3, x4, x5, x6], axis=1)


# %%
boatdataset = Sanren6_6(0.9)
boatdataset.set_dataset(batch_size=120)
# %%
boatdataset.model = RNN_Boat_Bert()
boatdataset.set_dataset(batch_size=120)
boatdataset.model_compile(learning_rate=2e-5)
# %%
boatdataset.start_training(epochs=100, weight_name='datas/best_sanren1_6')
# %%
boatdataset.model.load_weights('datas/best_sanren1_6')
# %%
pred = boatdataset.model.predict(boatdataset.valid)
y = boatdataset.y_valid

mx = np.expand_dims(np.max(pred, axis=2), 2)

win = ((pred - mx) == 0)*y
win = np.sum(win, axis=2)

print(np.sum(win[:, 0])/len(win))
print(np.sum(win[:, 1])/len(win))
print(np.sum(win[:, 2])/len(win))
print(np.sum(win[:, 3])/len(win))
print(np.sum(win[:, 4])/len(win))
print(np.sum(win[:, 5])/len(win))
# %%
