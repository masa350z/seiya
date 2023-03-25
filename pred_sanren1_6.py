# %%
from transformers import TFBertModel
from keras.layers import Dense
from keras.models import Model
import tensorflow as tf

from tqdm import tqdm
import numpy as np
import boatdata


class SanrenRNN3_6(boatdata.BoatDataset):
    """
    ラベルが「1着が誰か、2着、3着、、」のOne-Hot
    (x, 3, 6)次元のラベル
    """
    def __init__(self, ret_grade=True, sorted=True):
        super().__init__(ret_grade, sorted)
        self.set_label()

    def set_label(self):
        self.label = np.zeros((len(self.ar_num), 3, 6))
        th_ar = self.ret_sorted_th()[:, :3]

        for i in tqdm(range(len(self.ar_num))):
            for th in range(3):
                self.label[i][th][th_ar[i][th]-1] = 1

    def set_dataset(self, batch_size):
        x_train, x_valid, x_test = boatdata.split_data(self.tokenized_inputs)
        y_train, y_valid, y_test = boatdata.split_data(self.label)

        self.train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        self.valid = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size)
        self.test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

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


class RNN_Boat_Bert(Model):
    """
    BERTの出力をRNNデコーダーに通して1着、2着、3着それぞれが
    どの船なのか（6次元）をカテゴライズする
    出力：(x, 3, 6)
    """
    def __init__(self, bert_model='bert-base-uncased'):
        super(RNN_Boat_Bert, self).__init__(name='boat_bert')

        self.bert_model = TFBertModel.from_pretrained(bert_model)

        self.dense01 = Dense(512, activation='relu')
        self.dense02 = Dense(256, activation='relu')
        self.dense03 = Dense(128, activation='relu')

        self.output_01 = Dense(6, activation='softmax')
        self.output_02 = Dense(6, activation='softmax')
        self.output_03 = Dense(6, activation='softmax')

    def call(self, inputs):
        x = self.bert_model(inputs)[1]

        x1 = self.dense01(x)
        x_1 = self.dense02(x1)
        x_1 = self.dense03(x_1)
        x_1 = self.output_01(x_1)

        x2 = self.dense01(x1)
        x_2 = self.dense02(x2)
        x_2 = self.dense03(x_2)
        x_2 = self.output_02(x_2)

        x3 = self.dense01(x2)
        x_3 = self.dense02(x3)
        x_3 = self.dense03(x_3)
        x_3 = self.output_03(x_3)

        return tf.stack([x_1, x_2, x_3], axis=1)


# %%
boatdataset = SanrenRNN3_6()
boatdataset.model = RNN_Boat_Bert()
boatdataset.set_dataset(batch_size=120)
boatdataset.model_compile(learning_rate=2e-5)
# %%
boatdataset.start_training(epochs=100, weight_name='datas/best_sanren1_6')
# %%