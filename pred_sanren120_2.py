# %%
from transformers import TFBertModel
from keras.layers import Dense, Flatten ,concatenate
from keras.models import Model
import tensorflow as tf

from tqdm import tqdm
import numpy as np
import boatdata


class Sanren120(boatdata.BoatDataset):
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

            print('')


class RNN_Boat_Bert(Model):
    def __init__(self, bert_model='bert-base-uncased'):
        super(RNN_Boat_Bert, self).__init__(name='boat_bert')

        self.bert_model = TFBertModel.from_pretrained(bert_model)

        self.dense01 = Dense(256, activation='relu')
        self.conc01 = Dense(1024, activation='relu')

        self.output_01 = Dense(120, activation='softmax')

    def call(self, inputs):
        x, pre = inputs
        x= self.bert_model(x)[1]

        pre = self.dense01(tf.reshape(pre, (-1, 6*8)))

        x1 = self.conc01(concatenate([x, pre]))
        x1 = self.output_01(x1)

        return x1
    
    def ext_hidden(self, inputs):
        x, pre = inputs
        x= self.bert_model(x)[1]

        pre = self.dense01(tf.reshape(pre, (-1, 6*8)))

        x1 = self.conc01(concatenate([x, pre]))

        return x1    

# %%
boatdataset = Sanren120()
boatdataset.set_dataset(batch_size=120)
# %%
boatdataset.model = RNN_Boat_Bert()
boatdataset.set_dataset(batch_size=120)
boatdataset.model_compile(learning_rate=2e-5)
# %%
boatdataset.start_training(epochs=100, weight_name='datas/best_sanren120_2')
# %%
boatdataset.model.load_weights('datas/best_sanren120_2')
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