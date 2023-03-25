# %%
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, TFAlbertModel
from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
from tqdm import tqdm
# %%

def add_str(inp, str_):
    ret = []
    for i in inp:
        temp = [str_.format(str(j)) for j in i]
        ret.append(temp)
    ret = np.array(ret)

    return ret

def model_weights_random_init(model, init_ratio=0.0001):

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

def custom_loss(y_true, y_pred):
    loss = tf.multiply(y_true, y_pred)
    return tf.reduce_mean(loss)

class BoatData:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        df = pd.read_csv('data_combined.csv')

        ar_num = df[['no_1', 'no_2', 'no_3',
                     'no_4', 'no_5', 'no_6']]
        self.ar_num = np.array(ar_num, dtype='int16')

        ar_th = df[['1th', '2th', '3th',
                    '4th', '5th', '6th']]
        self.ar_th = np.array(ar_th, dtype='int16')

        ar_grade = df[['grade_1', 'grade_2', 'grade_3',
                       'grade_4', 'grade_5', 'grade_6']]
        self.ar_grade = np.array(ar_grade)
        self.ar_grade[self.ar_grade == '4'] = 'A1'
        self.ar_grade[self.ar_grade == '3'] = 'A2'
        self.ar_grade[self.ar_grade == '2'] = 'B1'
        self.ar_grade[self.ar_grade == '1'] = 'B2'

        ar_ze1 = df[['zenkoku_shoritshu_1', 'zenkoku_shoritshu_2',
                     'zenkoku_shoritshu_3', 'zenkoku_shoritshu_4',
                     'zenkoku_shoritshu_5', 'zenkoku_shoritshu_6']]
        self.ar_ze1 = np.round(np.array(ar_ze1, dtype='float16'), 0)

        ar_ze2 = df[['zenkoku_nirenritshu_1', 'zenkoku_nirenritshu_2',
                     'zenkoku_nirenritshu_3', 'zenkoku_nirenritshu_4',
                     'zenkoku_nirenritshu_5', 'zenkoku_nirenritshu_6']]
        self.ar_ze2 = np.round(np.array(ar_ze2, dtype='float16'), 0)

        ar_ze3 = df[['zenkoku_sanrenritshu_1', 'zenkoku_sanrenritshu_2',
                     'zenkoku_sanrenritshu_3', 'zenkoku_sanrenritshu_4',
                     'zenkoku_sanrenritshu_5', 'zenkoku_sanrenritshu_6']]
        self.ar_ze3 = np.round(np.array(ar_ze3, dtype='float16'), 0)

        ar_to1 = df[['tochi_shoritshu_1', 'tochi_shoritshu_2',
                     'tochi_shoritshu_3', 'tochi_shoritshu_4',
                     'tochi_shoritshu_5', 'tochi_shoritshu_6']]
        self.ar_to1 = np.round(np.array(ar_to1, dtype='float16'), 0)

        ar_to2 = df[['tochi_nirenritshu_1', 'tochi_nirenritshu_2',
                     'tochi_nirenritshu_3', 'tochi_nirenritshu_4',
                     'tochi_nirenritshu_5', 'tochi_nirenritshu_6']]
        self.ar_to2 = np.round(np.array(ar_to2, dtype='float16'), 0)

        ar_to3 = df[['tochi_sanrenritshu_1', 'tochi_sanrenritshu_2',
                     'tochi_sanrenritshu_3', 'tochi_sanrenritshu_4',
                     'tochi_sanrenritshu_5', 'tochi_sanrenritshu_6']]
        self.ar_to3 = np.round(np.array(ar_to3, dtype='float16'), 0)

        ar_incourse = df[['entry_course_1_no', 'entry_course_2_no',
                          'entry_course_3_no', 'entry_course_4_no',
                          'entry_course_5_no', 'entry_course_6_no']]
        self.ar_incourse = np.array(ar_incourse, dtype='int16')

        self.ar_field = np.array([str(int(ba))[-4:-2] for ba in df['index']])

        ar_condition = df[['weather',
                           'temperature',
                           'water_temperature',
                           'water_hight',
                           'wind_speed',
                           'wind_vector']]
        self.ar_condition = np.round(np.array(ar_condition,
                                              dtype='float16'), 1)

        indx = np.zeros(self.ar_th.shape)

        for i in range(6):
            for j in range(6):
                indx[:, j] += (self.ar_num[:, j] == self.ar_incourse[:, i])*i

        self.indx = indx.astype('int16')

        sanren_tan_col = df.columns[79:79+120]
        self.sanren_odds = df[sanren_tan_col]

    def ret_sorted(self, inp):
        ret = []
        if len(inp) == len(self.indx):
            for i in range(len(inp)):
                ret.append(inp[i][self.indx[i]])

            ret = np.array(ret)

            return ret
        else:
            raise ValueError("error!")

    def ret_field_conditions(self):
        fe = self.ar_field
        fe = np.array(['field{}'.format(i) for i in fe])
        cd = self.ar_condition.astype('str')

        ret = []
        for i in cd:
            temp = []
            temp.append('weath' + i[0])
            temp.append('temp' + i[1])
            temp.append('wtemp' + i[2])
            temp.append('whight' + i[3])
            temp.append('wspeed' + i[4])
            temp.append('wvect' + i[5])
            ret.append(temp)

        cd = np.array(ret)

        field_conditions = np.concatenate([fe.reshape(-1, 1), cd], axis=1, dtype='str')

        return field_conditions

    def ret_racers_data(self):
        num = self.ret_sorted(add_str(self.ar_num, 'num_{}'))
        ze1 = self.ret_sorted(add_str(self.ar_ze1, 'ze1_{}'))
        ze2 = self.ret_sorted(add_str(self.ar_ze2, 'ze2_{}'))
        ze3 = self.ret_sorted(add_str(self.ar_ze3, 'ze3_{}'))
        to1 = self.ret_sorted(add_str(self.ar_to1, 'to1_{}'))
        to2 = self.ret_sorted(add_str(self.ar_to2, 'to2_{}'))
        to3 = self.ret_sorted(add_str(self.ar_to3, 'to3_{}'))
        grade = self.ret_sorted(add_str(self.ar_grade, 'grade_{}'))

        num1 = np.concatenate([num[:, 0].reshape(-1, 1),
                               grade[:, 0].reshape(-1, 1),
                               ze1[:, 0].reshape(-1, 1),
                               ze2[:, 0].reshape(-1, 1),
                               ze3[:, 0].reshape(-1, 1),
                               to1[:, 0].reshape(-1, 1),
                               to2[:, 0].reshape(-1, 1),
                               to3[:, 0].reshape(-1, 1)], axis=1)

        num2 = np.concatenate([num[:, 1].reshape(-1, 1),
                               grade[:, 1].reshape(-1, 1),
                               ze1[:, 1].reshape(-1, 1),
                               ze2[:, 1].reshape(-1, 1),
                               ze3[:, 1].reshape(-1, 1),
                               to1[:, 1].reshape(-1, 1),
                               to2[:, 1].reshape(-1, 1),
                               to3[:, 1].reshape(-1, 1)], axis=1)

        num3 = np.concatenate([num[:, 2].reshape(-1, 1),
                               grade[:, 2].reshape(-1, 1),
                               ze1[:, 2].reshape(-1, 1),
                               ze2[:, 2].reshape(-1, 1),
                               ze3[:, 2].reshape(-1, 1),
                               to1[:, 2].reshape(-1, 1),
                               to2[:, 2].reshape(-1, 1),
                               to3[:, 2].reshape(-1, 1)], axis=1)

        num4 = np.concatenate([num[:, 3].reshape(-1, 1),
                               grade[:, 3].reshape(-1, 1),
                               ze1[:, 3].reshape(-1, 1),
                               ze2[:, 3].reshape(-1, 1),
                               ze3[:, 3].reshape(-1, 1),
                               to1[:, 3].reshape(-1, 1),
                               to2[:, 3].reshape(-1, 1),
                               to3[:, 3].reshape(-1, 1)], axis=1)

        num5 = np.concatenate([num[:, 4].reshape(-1, 1),
                               grade[:, 4].reshape(-1, 1),
                               ze1[:, 4].reshape(-1, 1),
                               ze2[:, 4].reshape(-1, 1),
                               ze3[:, 4].reshape(-1, 1),
                               to1[:, 4].reshape(-1, 1),
                               to2[:, 4].reshape(-1, 1),
                               to3[:, 4].reshape(-1, 1)], axis=1)

        num6 = np.concatenate([num[:, 5].reshape(-1, 1),
                               grade[:, 5].reshape(-1, 1),
                               ze1[:, 5].reshape(-1, 1),
                               ze2[:, 5].reshape(-1, 1),
                               ze3[:, 5].reshape(-1, 1),
                               to1[:, 5].reshape(-1, 1),
                               to2[:, 5].reshape(-1, 1),
                               to3[:, 5].reshape(-1, 1)], axis=1)

        racers_data = np.concatenate([num1, num2, num3,
                                      num4, num5, num6], axis=1)

        return racers_data

    def ret_inp_data(self, load=True):
        if load:
            return np.load('inp_data.npy')
        else:
            field_data = self.ret_field_conditions()

            seps = np.zeros(len(field_data)).astype('str')
            seps[seps == '0.0'] = '[SEP]'
            seps = seps.reshape(-1, 1)

            racers_data = self.ret_racers_data()

            half_len = int(len(field_data)*0.5)

            fdf = pd.DataFrame(field_data[:half_len].reshape(-1))
            fdf = np.array(fdf[~fdf.duplicated()]).reshape(-1).astype('str')

            rdf = pd.DataFrame(racers_data[:half_len].reshape(-1))
            rdf = np.array(rdf[~rdf.duplicated()]).reshape(-1).astype('str')

            n_tokens = np.concatenate([fdf, rdf])

            self.tokenizer.add_tokens(list(n_tokens), special_tokens=True)

            inp_data = np.concatenate(
                [field_data, seps, racers_data],
                axis=1)

            ret = []
            for j in tqdm(inp_data):
                token = self.tokenizer(list(j))['input_ids']
                temp = [101]
                for i in token:
                    if len(i) == 3:
                        temp.append(i[1])
                    else:
                        temp.append(100)
                temp.append(102)
                ret.append(temp)

            ret = np.array(ret)

            np.save('inp_data.npy', ret)

            return ret

    def ret_course_th(self):
        sort = self.ret_sorted(np.tile(
            np.array([1, 2, 3, 4, 5, 6]),
            (len(self.ar_num), 1)))

        th1 = sort.reshape(-1, 1)[(self.ar_th[:, 0]-1)]
        th2 = sort.reshape(-1, 1)[(self.ar_th[:, 1]-1)]
        th3 = sort.reshape(-1, 1)[(self.ar_th[:, 2]-1)]

        course_th = np.concatenate([th1, th2, th3], axis=1)

        return course_th

    def ret_course_th_onehot(self):
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

        y_data = np.zeros((len(self.ar_num), 120))
        th_ar = self.ret_course_th()

        for i in tqdm(range(len(th_ar))):
            temp_ar = th_ar[i]

            sanren = ''
            for nm in temp_ar:
                sanren += str(nm)

            y_data[i][sanren_dic[sanren]] = 1
        
        return y_data
    
    def ret_course_th_123(self):
        y_data = np.zeros((len(self.ar_num), 3, 6))
        th_ar = self.ret_course_th()

        for i in tqdm(range(len(self.ar_num))):
            for th in range(3):
                y_data[i][th][th_ar[i][th]-1] = 1
        
        return y_data

    def ret_sanren_odds(self):
        sanren_odds_lis = []
        for i in tqdm(range(len(self.ar_th))):
            odds = self.sanren_odds.iloc[i]['sanren_tan_{}{}{}'.format(self.ar_th[i][0], self.ar_th[i][1], self.ar_th[i][2])]
            sanren_odds_lis.append(odds)
        
        return np.array(sanren_odds_lis)
# %%
boatdata = BoatData()
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
        self.bert_model = TFBertModel.from_pretrained(bert_model)
        self.output_layer = Dense(num_classes, activation='softmax')
    
    def call(self, inputs):
        x = self.bert_model(inputs)[1]
        x = self.output_layer(x)
        
        return x

class Odds_Attention(Model):
    def __init__(self, bert_model='bert-base-uncased',
                 compile=False, trainable=False):
        super(Odds_Attention, self).__init__(name='boat_bert')
        self.bert = TFBertModel.from_pretrained(bert_model)
        self.attention_output = Dense(120, activation='softmax')

    def call(self, inputs):
        x = self.bert(inputs)[1]
        x = self.attention_output(x)

        return x
# %%
X = boatdata.ret_inp_data()
Y = boatdata.ret_course_th_onehot()
#Y = boatdata.ret_course_th_123()
X[X > 10000] -= 10000
mask = np.array([True, True, True, True, True, True, True, True, True,
                 True, True, False, False, False, False, False, False,
                 True, True, False, False, False, False, False, False,
                 True, True, False, False, False, False, False, False,
                 True, True, False, False, False, False, False, False,
                 True, True, False, False, False, False, False, False,
                 True, True, False, False, False, False, False, False,
                 True])

X = X[:,mask]
# %%
train_len = int(len(X)*0.7)

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
batch_size = 25
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
# %%
model = Odds_Attention(bert_model='bert-large-uncased')
LEARNING_RATE = 1e-4  # 学習率
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# %%
model.fit(train_dataset)
# %%
best_weights = None

epochs = 1000

best_val_loss = float('inf')
k_freeze = 5
freeze = k_freeze

val_loss, val_acc = model.evaluate(valid_dataset)
print(f"Initial valid loss: {val_loss}, initial valid accuracy: {val_acc}")

# 学習を開始する
for epoch in range(epochs):
    # トレーニングデータを使用してモデルを学習
    model.fit(train_dataset)

    # valid dataでモデルを評価し、valid lossを取得
    val_loss, val_acc = model.evaluate(x_valid, y_valid)

    # valid lossが減少した場合、重みを保存
    if val_loss < best_val_loss:
        freeze = 0
        best_val_loss = val_loss
        best_weights = model.get_weights()
        print(f"Epoch {epoch + 1}: Valid loss decreased to {val_loss}, saving weights.")

    # valid lossが減少しなかった場合、保存しておいた最良の重みをロード
    else:
        if freeze == 0:
            model.set_weights(best_weights)
            model = model_weights_random_init(model)
            freeze = k_freeze
            print(f"Epoch {epoch + 1}: Valid loss did not decrease, loading best weights.")
        else:
            print(f"Epoch {epoch + 1}: Valid loss did not decrease.")
    
    freeze = freeze - 1 if freeze > 0 else freeze
    
    print('')
# %%
model.set_weights(best_weights)
# %%
model.evaluate(test_dataset)
# %%
model.save_weights('boat_bert_weigths')
# %%
pred = model.predict(X, batch_size=120)
# %%
np.save('pred_sanren.npy', pred)
# %%
