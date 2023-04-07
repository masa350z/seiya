# %%
from transformers import TFBertModel
from keras.layers import Dense, concatenate
from keras.models import Model
import tensorflow as tf

from tqdm import tqdm
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

            y_ = np.zeros((6,6))
            for j in range(6):
                y_[j][co_to_th[j]] = 1
            
            data_y.append(y_)

        self.label = np.array(data_y)

    def set_dataset(self, batch_size):
        self.x_train, self.x_valid, self.x_test = boatdata.split_data(self.tokenized_inputs)
        self.y_train, self.y_valid, self.y_test = boatdata.split_data(self.label)

        self.train = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).batch(batch_size)
        self.valid = tf.data.Dataset.from_tensor_slices((self.x_valid, self.y_valid)).batch(batch_size)
        self.test = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(batch_size)

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

            mx = np.expand_dims(np.max(pred,axis=2), 2)

            win = ((pred - mx)==0)*y
            win = np.sum(win,axis=2)

            print(np.sum(win[:,0])/len(win))
            print(np.sum(win[:,1])/len(win))
            print(np.sum(win[:,2])/len(win))
            print(np.sum(win[:,3])/len(win))
            print(np.sum(win[:,4])/len(win))
            print(np.sum(win[:,5])/len(win))

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

        self.output_01 = Dense(6, activation='softmax')
        self.output_02 = Dense(6, activation='softmax')
        self.output_03 = Dense(6, activation='softmax')
        self.output_04 = Dense(6, activation='softmax')
        self.output_05 = Dense(6, activation='softmax')
        self.output_06 = Dense(6, activation='softmax')

    def call(self, inputs):
        x = self.bert_model(inputs)

        x1 = self.dense01(x[0][:,9])
        x_1 = self.output_01(x1)

        x2 = self.dense02(x[0][:,11])
        x_2 = self.output_02(x2)

        x3 = self.dense03(x[0][:,13])
        x_3 = self.output_03(x3)

        x4 = self.dense04(x[0][:,15])
        x_4 = self.output_04(x4)

        x5 = self.dense05(x[0][:,17])
        x_5 = self.output_05(x5)

        x6 = self.dense06(x[0][:,19])
        x_6 = self.output_06(x6)

        return tf.stack([x_1, x_2, x_3, x_4, x_5, x_6], axis=1)


# %%
boatdataset = boatdata.BoatDataset()
# %%
boatdataset.label
# %%
boatdataset = Sanren6_6()
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

mx = np.expand_dims(np.max(pred,axis=2), 2)

win = ((pred - mx)==0)*y
win = np.sum(win,axis=2)

print(np.sum(win[:,0])/len(win))
print(np.sum(win[:,1])/len(win))
print(np.sum(win[:,2])/len(win))
print(np.sum(win[:,3])/len(win))
print(np.sum(win[:,4])/len(win))
print(np.sum(win[:,5])/len(win))
# %%
(np.sum(win[:,0])/len(win))*(np.sum(win[:,1])/len(win))*(np.sum(win[:,2])/len(win))