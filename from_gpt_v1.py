# %%
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, TFAlbertModel
from tensorflow.keras.layers import Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
from tqdm import tqdm
# %%
# ハイパーパラメーターの設定
MAX_SEQ_LEN = 7+2  # シーケンスの最大長
NUM_CLASSES = 6  # カテゴリの数
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
class Attention(Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.d1 = Dense(units, activation='relu')
        self.d2 = Dense(units, activation='relu')
        self.d3 = Dense(units, activation='sigmoid')

    def call(self, inputs):
        x = self.d1(inputs)
        x = self.d2(x)
        x = self.d3(x)
        
        return inputs*x

class Boat_Bert(Model):
    def __init__(self, num_classes, bert_model='bert-base-uncased'):
        super(Boat_Bert, self).__init__(name='boat_bert')
        self.attention = Attention(768)
        self.bert_model = TFBertModel.from_pretrained(bert_model)
        self.output_layer = Dense(num_classes, activation='softmax')
    
    def call(self, inputs):
        x = self.bert_model(inputs)[1]
        x = self.output_layer(self.attention(x))
        
        return x

class Boat_Albert(Model):
    def __init__(self, num_classes, albert_model='albert-base-v2'):
        super(Boat_Albert, self).__init__(name='boat_albert')

        self.albert_model = TFAlbertModel.from_pretrained(albert_model, output_hidden_states=True, output_attentions=True)  # load ALBERT model
        self.dropout = Dropout(0.2)
        self.output_layer = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.albert_model(inputs)
        #x = self.dropout(x)
        x = self.output_layer(x[0][:, 0, :])
        
        return x

class Encoder(Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.bert = TFBertModel.from_pretrained('bert-base-uncased')
        #self.bert = TFBertModel.from_pretrained('bert-large-uncased')
        #self.bert = TFAlbertModel.from_pretrained('albert-base-v2', output_hidden_states=True, output_attentions=True)  # load ALBERT model

    def call(self, inputs):
        bert_outputs = self.bert(inputs)[1]  # only use the last layer
        #bert_outputs = self.bert(inputs)[0][:, 0, :]  # only use the last layer
        return bert_outputs

class Decoder(Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention1 = Attention(768)
        self.attention2 = Attention(768)
        self.attention3 = Attention(768)

        self.attention1_1 = Attention(512)
        self.attention2_1 = Attention(512)
        self.attention3_1 = Attention(512)

        self.pred_1th = Dense(1024, activation='relu')
        self.pred_2th = Dense(1024, activation='relu')
        self.pred_3th = Dense(1024, activation='relu')

        self.pred_1th_1 = Dense(512, activation='relu')
        self.pred_2th_1 = Dense(512, activation='relu')
        self.pred_3th_1 = Dense(512, activation='relu')

        self.out_1th = Dense(6, activation='softmax')
        self.out_2th = Dense(6, activation='softmax')
        self.out_3th = Dense(6, activation='softmax')

    def call(self, inputs):
        outputs = []

        inp1 = self.attention1(inputs)
        x1 = self.pred_1th(inp1)
        x1_1 = self.pred_1th_1(x1)
        x1_2 = self.attention1_1(self.pred_1th_2(x1_1))
        x1_3 = self.pred_1th_3(x1_2)
        x1_4 = self.pred_1th_4(x1_3)
        x1_5 = self.attention1_2(self.pred_1th_5(x1_4))

        inp2 = self.attention2(inputs)
        x2 = self.pred_2th(concatenate([x1_1, inp2]))
        x2_1 = self.pred_2th_1(x2)
        x2_2 = self.attention2_1(self.pred_2th_2(x2_1))
        x2_3 = self.pred_2th_3(x2_2)
        x2_4 = self.pred_2th_4(x2_3)
        x2_5 = self.attention2_2(self.pred_2th_5(x2_4))

        inp3 = self.attention3(inputs)
        x3 = self.pred_2th(concatenate([x2_1, inp3]))
        x3_1 = self.pred_3th_1(x3)
        x3_2 = self.attention3_1(self.pred_3th_2(x3_1))
        x3_3 = self.pred_3th_3(x3_2)
        x3_4 = self.pred_3th_4(x3_3)
        x3_5 = self.attention3_2(self.pred_3th_5(x3_4))
        
        outputs.append(self.out_1th(x1_5))
        outputs.append(self.out_2th(x2_5))
        outputs.append(self.out_3th(x3_5))
        
        outputs = tf.stack(outputs, axis=1)

        return outputs
    
class EncoderDecoder(Model):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, inputs):
        enc_out = self.encoder(inputs)
        dec_out = self.decoder(enc_out)
        return dec_out

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

X = np.array(inp_data)
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

y_data = np.zeros((len(bt_data), 120))
th_ar = np.array(bt_data[['1th', '2th', '3th']], dtype='int32')

for i in tqdm(range(len(bt_data))):
    temp_ar = th_ar[i]

    sanren = ''
    for nm in temp_ar:
        sanren += str(nm)

    y_data[i][sanren_dic[sanren]] = 1
# %%
y_data = np.zeros((len(bt_data), 3, 6))
th_ar = np.array(bt_data[['1th', '2th', '3th']], dtype='int32')

for i in tqdm(range(len(bt_data))):
    for th in range(3):
        y_data[i][th][th_ar[i][th]-1] = 1
# %%
y_data = y_data[:,0]
# %%
y_data = np.array(bt_data[['1th', '2th', '3th']], dtype='int32')
# %%
np.save('x_train.npy', X)
# %%
X = np.load('x_train.npy')
# %%
#X[:,1:-1] -= 10000
#X[X > 10000] -= 10000
Y = y_data[:len(X)]
# %%
ind = np.arange(len(X))
np.random.shuffle(ind)

X = X[ind]
Y = Y[ind]
# %%
x_train = X[:train_len]
y_train = Y[:train_len]

x_test = X[train_len:]
y_test = Y[train_len:]

valid_len = int(len(x_train)*0.2)
x_valid = x_train[-valid_len:]
y_valid = y_train[-valid_len:]

x_train = x_train[:-valid_len]
y_train = y_train[:-valid_len]
# %%
batch_size = 120
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
# %%
#checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=True)
#model.fit(x=x_train, y=y_train, epochs=10, batch_size=360, validation_split=0.2, callbacks=[checkpoint])
# %%
model = EncoderDecoder()
#model = Boat_Albert(120)

checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
# %%
model.fit(x_train, y_train, epochs=100, batch_size=720, validation_split=0.2, callbacks=[checkpoint])
# %%
sa = model.predict(x_test)
# %%
model.load_weights('best_model.h5')
# %%
pred = model.predict(x_test)
# %%
np.sum(((pred - np.max(pred, axis=1).reshape(-1, 1)) == 0) * y_test)/len(y_test)
# %%
mx = (np.max(sa,axis=2).reshape(len(sa),3,1))
pred_onehot = (sa - mx) == 0
# %%
pred123 = pred_onehot*y_test
pred123 = np.sum(pred123,axis=2)
# %%
np.sum((np.sum(pred123,axis=1)==3))/len(pred123)
# %%
np.sum((np.sum(pred123[:,:2],axis=1)==2))/len(pred123)
# %%
np.sum((np.sum(pred123[:,:1],axis=1)==1))/len(pred123)
#%%
def model_weights_random_init(model, init_ratio=0.0001):
    """
    モデルの重みをランダムに初期化する関数
    Args:
        model: tf.keras.Model, モデル
        init_ratio: float, 重みをランダム初期化する割合
    Returns:
        tf.keras.Model, 重みを一部ランダム初期化したモデル
    """
    # モデルの重みを取得する
    weights = model.get_weights()

    # 重みをランダムに初期化する
    for i, weight in enumerate(weights):
        if len(weight.shape) == 2:
            # 重み行列の場合、init_ratioの割合でランダム初期化する
            rand_mask = np.random.binomial(1, init_ratio, size=weight.shape)
            rand_weights = np.random.randn(*weight.shape) * rand_mask
            weights[i] = weight * (1 - rand_mask) + rand_weights

    # モデルの重みをセットする
    model.set_weights(weights)

    return model
# %%
# 最良の重みを保存するための変数
best_weights = None

# エポック数
epochs = 1000

# valid dataのlossを監視するための変数
best_val_acc = 0
freeze = 0
k_freeze = 10

# 学習を開始する前に、最初にvalid dataを評価する
val_loss, val_acc = model.evaluate(valid_dataset)
print(f"Initial valid loss: {val_loss}, initial valid accuracy: {val_acc}")

# 学習を開始する
for epoch in range(epochs):
    # トレーニングデータを使用してモデルを学習
    model.fit(train_dataset)

    # valid dataでモデルを評価し、valid lossを取得
    val_loss, val_acc = model.evaluate(valid_dataset)

    # valid lossが減少した場合、重みを保存
    if val_acc > best_val_acc:
        freeze = 0
        best_val_acc = val_acc
        best_weights = model.get_weights()
        print(f"Epoch {epoch + 1}: Valid acc increased to {val_acc}, saving weights.")

    # valid lossが減少しなかった場合、保存しておいた最良の重みをロード
    else:
        if freeze == 0:
            model.set_weights(best_weights)
            model = model_weights_random_init(model)
            freeze = k_freeze
            print(f"Epoch {epoch + 1}: Valid acc did not increase, loading best weights.")
        else:
            print(f"Epoch {epoch + 1}: Valid acc did not increase.")
    
    freeze = freeze - 1 if freeze > 0 else freeze
    
    print('')
# %%
model.set_weights(best_weights)
# %%
model.evaluate(valid_dataset)