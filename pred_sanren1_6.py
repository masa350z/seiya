# %%
from transformers import TFBertModel
from keras.layers import Dense, concatenate
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
        self.dense02 = Dense(512, activation='relu')
        self.dense03 = Dense(512, activation='relu')

        self.output_01 = Dense(6, activation='softmax')
        self.output_02 = Dense(6, activation='softmax')
        self.output_03 = Dense(6, activation='softmax')

    def call(self, inputs):
        x = self.bert_model(inputs)[1]

        x1 = self.dense01(x)
        x_1 = self.output_01(x1)

        x2 = self.dense02(concatenate([x1, x]))
        x_2 = self.output_02(x2)

        x3 = self.dense03(concatenate([x2, x]))
        x_3 = self.output_03(x3)

        return tf.stack([x_1, x_2, x_3], axis=1)


# %%
boatdataset = SanrenRNN3_6()
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
# %%
np.sum(np.sum(np.abs(y[:,0] - ((pred[:,0] - np.max(pred[:,0],axis=1).reshape(-1,1)) == 0)*1),axis=1)==0)/len(y)
# %%
np.sum(np.sum(np.abs(y[:,1] - ((pred[:,1] - np.max(pred[:,1],axis=1).reshape(-1,1)) == 0)*1),axis=1)==0)/len(y)
# %%
np.sum(np.sum(np.abs(y[:,2] - ((pred[:,2] - np.max(pred[:,2],axis=1).reshape(-1,1)) == 0)*1),axis=1)==0)/len(y)
# %%
0.5638*0.2838*0.2312
# %%
np.sum((np.sum(np.sum(np.abs(y[:,:2] - (pred[:,:2] - np.max(pred[:,:2], axis=2).reshape(-1,2,1) == 0)*1),axis=2),axis=1))==0)/len(y)
# %%
np.sum((np.sum(np.sum(np.abs(y[:,:3] - (pred[:,:3] - np.max(pred[:,:3], axis=2).reshape(-1,3,1) == 0)*1),axis=2),axis=1))==0)/len(y)
# %%
