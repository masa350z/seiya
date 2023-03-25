# %%
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, TFAlbertModel
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
from tqdm import tqdm
# %%
# ハイパーパラメーターの設定
MAX_SEQ_LEN = 7+2  # シーケンスの最大長
NUM_CLASSES = 120  # カテゴリの数
LEARNING_RATE = 2e-5  # 学習率
# %%
# 自前の単語を含めたトークナイザーの作成
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bt_data = pd.read_csv('boatrace_data.csv')
#bt_data = bt_data[:40000]
bt_data = bt_data[['Unnamed: 0', 'no_1', 'no_2', 'no_3', 'no_4', 'no_5', 'no_6', '1th', '2th', '3th' , '4th', '5th', '6th']]

train_len = int(len(bt_data)*0.7)
# %%
numbers = bt_data[['no_1', 'no_2', 'no_3', 'no_4', 'no_5', 'no_6']][:int(len(bt_data)*0.5)]
numbers = np.array(numbers, dtype='int32')
numbers = numbers.reshape(-1)
numbers = pd.DataFrame(numbers)
numbers = np.array(numbers[~numbers.duplicated()]).reshape(-1)

tokenizer.add_tokens(['卍卍{}卍卍'.format(i) for i in numbers], special_tokens=True)
tokenizer.add_tokens(['卍卍{}卍卍'.format(str(i).zfill(2)) for i in range(1,25)], special_tokens=True)
# %%
# BERTモデルの読み込みと転移学習用のカスタムレイヤーの追加
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
#bert_model = TFBertModel.from_pretrained('bert-large-uncased')
input_layer = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='input_ids')
bert_layer = bert_model(input_layer)[1]
#bert_layer = bert_model(input_layer)[0][:,1]
output_layer = Dense(NUM_CLASSES, activation='softmax')(bert_layer)
model = Model(inputs=input_layer, outputs=output_layer)
# %%
class Boat_Bert(Model):
    def __init__(self, num_classes, bert_model='bert-base-uncased'):
        super(Boat_Bert, self).__init__(name='boat_bert')

        self.bert_model = TFBertModel.from_pretrained(bert_model)
        self.dropout = Dropout(0.2)
        self.output_layer = Dense(num_classes, activation='softmax')
    
    def call(self, inputs):
        x = self.bert_model(inputs)
        #x = self.dropout(x)
        x = self.output_layer(x[1])
        
        return x
# %%
albert_model = TFAlbertModel.from_pretrained('albert-base-v2', output_hidden_states=True, output_attentions=True)  # load ALBERT model

input_layer = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='input_ids')
albert_layer = albert_model(input_layer)[0][:, 0, :]  # use the output of the first token [CLS] as input to the classification layer
output_layer = Dense(NUM_CLASSES, activation='softmax')(albert_layer)
model = Model(inputs=input_layer, outputs=output_layer)
# %%
# モデルのコンパイル
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# %%
sanren_dic = {}
count = 0
for i in range(6):
    for j in range(6):
        for k in range(6):
            j1 = i == j
            j2 = j == k
            j3 = i == k
            if not (j1 or j2 or j3):
                sanren_dic['{}{}{}'.format(i+1, j+1, k+1)] = count
                count += 1
# %%
y_data = np.zeros((len(bt_data), 120))
th_ar = np.array(bt_data[['1th', '3th', '2th']], dtype='int32')
inp_data = []
for i in tqdm(range(len(bt_data))):
#for i in tqdm(range(100000)):
    temp_df = bt_data.iloc[i]

    ba = str(int(temp_df['Unnamed: 0']))[-4:-2]
    no1 = str(int(temp_df['no_1']))
    no2 = str(int(temp_df['no_2']))
    no3 = str(int(temp_df['no_3']))
    no4 = str(int(temp_df['no_4']))
    no5 = str(int(temp_df['no_5']))
    no6 = str(int(temp_df['no_6']))

    inp_temp = [101]
    tokens = tokenizer(['卍卍{}卍卍'.format(ba) for ba in [ba, no1, no2, no3, no4, no5, no6]])['input_ids']
    for token in tokens:
        if len(token) == 3:
            inp_temp.append(token[1])
        else:
            inp_temp.append(100)
    inp_temp.append(102)
    inp_temp.append(102)
    inp_data.append(inp_temp)

    temp_ar = th_ar[i]

    sanren = ''
    for nm in temp_ar:
        sanren += str(nm)

    y_data[i][sanren_dic[sanren]] = 1
# %%
ba_data = [str(int(ba))[-4:-2] for ba in bt_data['Unnamed: 0']]
no_data = [[str(int(temp_df[f'no_{j}'])) for j in range(1, 7)] for _, temp_df in bt_data.iterrows()]

inp_data = []

batch_size = 1000
for i in tqdm(range(0, len(bt_data), batch_size)):
    temp_ba_data = ba_data[i:i+batch_size]
    temp_no_data = no_data[i:i+batch_size]

    ba_str = [f'卍卍{ba}卍卍' for ba in temp_ba_data]
    no_str = [[f'卍卍{n}卍卍' for n in no] for no in temp_no_data]
    texts = [''.join([ba, *no]) for ba, no in zip(ba_str, no_str)]

    tokens = tokenizer.batch_encode_plus(texts, padding='max_length', max_length=MAX_SEQ_LEN, truncation=True)['input_ids']
    inp_data.extend(tokens)
# %%
y_data = np.zeros((len(bt_data), 120))
th_ar = np.array(bt_data[['1th', '2th', '3th']], dtype='int32')

for i in tqdm(range(len(bt_data))):
    temp_ar = th_ar[i]

    sanren = ''
    for nm in temp_ar:
        sanren += str(nm)

    y_data[i][sanren_dic[sanren]] = 1
# %%
y_data = np.zeros((len(bt_data), 6))
th_ar = np.array(bt_data[['1th', '2th', '3th', '4th', '5th', '6th']], dtype='int32')
th1 = th_ar[:,0]-1

for i in tqdm(range(len(bt_data))):
    ar = th1[i]

    y_data[i][ar] = 1
# %%

# %%
X = np.array(inp_data)
# %%
X = np.load('x_train.npy')
X[:,1:-1] -= 10000
Y = y_data[:len(X)]

x_train = X[:train_len]
y_train = Y[:train_len]

x_test = X[train_len:]
y_test = Y[train_len:]
# %%
# モデル保存用コールバック
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=True)

# モデルの訓練
model.fit(x=x_train, y=y_train, epochs=10, batch_size=360, validation_split=0.2, callbacks=[checkpoint])
# %%
model.load_weights('best_model.h5')
# %%
pred = model.predict(x_test[:100])
pred
# %%
np.sum(((pred - np.max(pred, axis=1).reshape(-1, 1)) == 0) * y_test)/len(y_test)
# %%
df = pd.read_csv('boatrace_data.csv')
# %%
list(df.columns)
# %%
df[['weather','temperature','water_temperature','water_hight','wind_speed','wind_vector']]
# %%
model = Boat_Bert(6)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# %%
a(x_test[:2])
# %%
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
# %%
class CustomBertModel(tf.keras.Model):
    def __init__(self, num_labels):
        super(CustomBertModel, self).__init__()
        self.bert = TFBertModel.from_pretrained('bert-base-uncased')
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.dense = tf.keras.layers.Dense(num_labels, activation='softmax')

    def call(self, inputs):
        outputs = self.bert(inputs)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        outputs = self.dense(pooled_output)
        return outputs
# %%
aa = CustomBertModel(7)
# %%
x_train[:10]