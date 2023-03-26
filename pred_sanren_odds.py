# %%
from transformers import TFBertModel
from keras.layers import Dense, concatenate
from keras.models import Model
import tensorflow as tf

from tqdm import tqdm
import numpy as np
import boatdata


def custom_loss(y_true, y_pred):
    loss = tf.multiply(y_true, y_pred)

    return tf.reduce_mean(tf.reduce_sum(loss, axis=1)) + 1


class SanrenOdds(boatdata.BoatDataset):
    """
    ラベルが3連単の組み合わせ120通りのデータセット
    (x, 120)次元のラベル
    """
    def __init__(self, ret_grade=True, sorted=True):
        super().__init__(ret_grade, sorted)
        self.seikai_odds = self.ret_sanren_odds()
        self.odds = self.ret_sorted_odds()
        self.pred_sanren = np.load('datas/pred_sanren.npy')
        self.exp = self.odds*self.pred_sanren

    def set_label(self, penalty):
        label = np.zeros((len(self.ar_num), 120))
        th_ar = self.ret_sorted_th()[:, :3]

        for i in tqdm(range(len(th_ar))):
            temp_ar = th_ar[i]

            sanren = ''
            for nm in temp_ar:
                sanren += str(nm)

            label[i][self.sanren_dic[sanren]] = 1

        label = label*self.seikai_odds.reshape(-1, 1)
        label = np.where(label == 0, 1, -(label-1))
        label = np.pad(label, (0, 1))[:-1]
        label = np.where(label == 0, penalty, label)

        self.label = label

    def set_dataset(self, batch_size):
        self.x_tr, self.x_vl, self.x_te = boatdata.split_data(self.tokenized_inputs)
        self.exp_tr, self.exp_vl, self.exp_te = boatdata.split_data(self.exp)
        self.y_tr, self.y_vl, self.y_te = boatdata.split_data(self.label)

        tr = tf.data.Dataset.from_tensor_slices((self.x_tr, self.exp_tr))
        tr_ = tf.data.Dataset.from_tensor_slices(self.y_tr)
        self.tr = tf.data.Dataset.zip((tr, tr_)).batch(batch_size)

        val = tf.data.Dataset.from_tensor_slices((self.x_vl, self.exp_vl))
        val_ = tf.data.Dataset.from_tensor_slices(self.y_vl)
        self.val = tf.data.Dataset.zip((val, val_)).batch(batch_size)

        te = tf.data.Dataset.from_tensor_slices((self.x_te, self.exp_te))
        te_ = tf.data.Dataset.from_tensor_slices(self.y_te)
        self.te = tf.data.Dataset.zip((te, te_)).batch(batch_size)

    def model_compile(self, learning_rate=False):
        if learning_rate:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = 'Adam'

        self.model.compile(optimizer=optimizer,
                           loss=custom_loss)

    def start_training(self, epochs, weight_name, k_freeze=3):
        best_val_loss = float('inf')
        k_freeze = 3
        freeze = k_freeze

        val_loss = self.model.evaluate(self.val)
        print(f"Initial valid loss: {val_loss}")

        # 学習を開始する
        for epoch in range(epochs):
            self.model.fit(self.tr)
            val_loss = self.model.evaluate(self.val)

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


class ConfAttention(Model):
    def __init__(self):
        super(ConfAttention, self).__init__(name='confattention')
        self.dense01 = Dense(2048, activation='relu')
        self.dense02 = Dense(1024, activation='relu')
        self.dense03 = Dense(512, activation='relu')
        self.conf_attention = Dense(120, activation='relu')

    def call(self, inputs):
        x = self.dense01(inputs)
        x = self.dense02(x)
        x = self.dense03(x)
        x = self.conf_attention(x)

        return x


class Odds_Boat_Bert(Model):
    """
    BERTの出力をそのまま
    （クラス数）次元のカテゴライズレイヤーに渡すモデル
    """
    def __init__(self, bert_model='bert-base-uncased'):
        super(Odds_Boat_Bert, self).__init__(name='boat_bert')
        self.bert_model = TFBertModel.from_pretrained(bert_model)
        self.attention = ConfAttention()
        self.concat_layer = Dense(2048, activation='relu')
        self.dense01 = Dense(1024, activation='relu')
        self.dense02 = Dense(512, activation='relu')
        self.output_layer = Dense(121, activation='softmax')

    def call(self, inputs):
        x, exp = inputs
        x = self.bert_model(x)[1]
        conc = concatenate([x, exp])
        x = self.concat_layer(conc)
        x = self.dense01(x)
        x = self.dense02(x)
        # attention = self.attention(conc)

        x = self.output_layer(x)

        return x


# %%
bt = SanrenOdds()
bt.set_label(0.2)
bt.model = Odds_Boat_Bert()
bt.set_dataset(batch_size=120)
bt.model_compile(learning_rate=2e-5)
# %%
bt.start_training(epochs=100, weight_name='datas/sanren_odds/bert_sanren120')
# %%
bt.model.load_weights('datas/sanren_odds/bert_sanren120')
# %%
bt.model(bt.tr)
# %%
