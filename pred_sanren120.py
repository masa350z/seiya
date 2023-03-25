# %%
from transformers import TFBertModel, TFAlbertModel
from keras.layers import Dense
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


class Simple_Boat_Bert(Model):
    """
    BERTの出力をそのまま
    （クラス数）次元のカテゴライズレイヤーに渡すモデル
    """
    def __init__(self, num_classes, bert_model='bert-base-uncased'):
        super(Simple_Boat_Bert, self).__init__(name='boat_bert')

        self.bert_model = TFBertModel.from_pretrained(bert_model)
        self.output_layer = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.bert_model(inputs)[1]
        x = self.output_layer(x)

        return x


class Simple_Boat_Albert(Model):
    def __init__(self, num_classes, albert_model='albert-base-v2'):
        super(Simple_Boat_Albert, self).__init__(name='boat_albert')

        self.albert_model = TFAlbertModel.from_pretrained(albert_model,
                                                          output_hidden_states=True,
                                                          output_attentions=True)
        self.output_layer = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.albert_model(inputs)
        x = self.output_layer(x[0][:, 0, :])

        return x


# %%
boatdataset = Sanren120()
boatdataset.model = Simple_Boat_Bert(120)
boatdataset.set_dataset(batch_size=120)
boatdataset.model_compile(learning_rate=2e-5)
# %%
boatdataset.start_training(epochs=100, weight_name='datas/best_sanren120')
# %%
