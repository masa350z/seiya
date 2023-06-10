# %%
from transformers import BertTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np
import os


def split_data(inp, tr_rate=0.6, val_rate=0.2):
    train_len = int(len(inp)*tr_rate)
    valid_len = int(len(inp)*val_rate)

    train = inp[:train_len]
    valid = inp[train_len:train_len+valid_len]
    test = inp[train_len+valid_len:]

    return train, valid, test


def ret_sanren():
    sanren = []
    for i in range(6):
        for j in range(6):
            for k in range(6):
                c1 = i == j
                c2 = i == k
                c3 = j == k
                if not (c1 or c2 or c3):
                    sanren.append((i+1)*100 + (j+1)*10 + (k+1)*1)

    return np.array(sanren)


def ret_niren():
    niren = []
    for i in range(6):
        for j in range(6):
            if not i == j:
                niren.append((i+1)*10 + (j+1)*1)

    return np.array(niren)


class BoatDataset:
    def __init__(self, race_field=None):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.df = pd.read_csv('datas/boatdata.csv')
        self.ar_field = np.array([str(int(ba))[-4:-2] for ba in self.df['index']], dtype='int16')
        if race_field:
            self.df = self.df[self.ar_field == race_field].reset_index(drop=True)
            self.ar_field = np.array([str(int(ba))[-4:-2] for ba in self.df['index']], dtype='int16')

        self.read_boatcsv()
        
        self.sanren_indx = ret_sanren()
        self.sanren_odds = self.ret_all_sanren_odds()
        self.sanren_odds = np.where(self.sanren_odds == 0, 1, self.sanren_odds)

        self.niren_indx = ret_niren()
        self.niren_odds = self.ret_all_niren_odds()
        self.niren_odds = np.where(self.niren_odds == 0, 1, self.niren_odds)

    def read_boatcsv(self):
        """
        boatdata.csv の内容を自己の変数に読み込み
        """
        ar_num = self.df[['no_1', 'no_2', 'no_3',
                          'no_4', 'no_5', 'no_6']]
        self.ar_num = np.array(ar_num, dtype='int16')

        ar_th = self.df[['1th', '2th', '3th',
                         '4th', '5th', '6th']]
        self.ar_th = np.array(ar_th, dtype='int16')

        ar_grade = self.df[['grade_1', 'grade_2', 'grade_3',
                            'grade_4', 'grade_5', 'grade_6']]
        self.ar_grade_num = np.array(ar_grade).astype('int16')

        ar_ze1 = self.df[['zenkoku_shoritshu_1', 'zenkoku_shoritshu_2',
                          'zenkoku_shoritshu_3', 'zenkoku_shoritshu_4',
                          'zenkoku_shoritshu_5', 'zenkoku_shoritshu_6']]
        self.ar_ze1 = np.round(np.array(ar_ze1, dtype='float16'), 0)

        ar_ze2 = self.df[['zenkoku_nirenritshu_1', 'zenkoku_nirenritshu_2',
                          'zenkoku_nirenritshu_3', 'zenkoku_nirenritshu_4',
                          'zenkoku_nirenritshu_5', 'zenkoku_nirenritshu_6']]
        self.ar_ze2 = np.round(np.array(ar_ze2, dtype='float16'), 0)

        ar_ze3 = self.df[['zenkoku_sanrenritshu_1', 'zenkoku_sanrenritshu_2',
                          'zenkoku_sanrenritshu_3', 'zenkoku_sanrenritshu_4',
                          'zenkoku_sanrenritshu_5', 'zenkoku_sanrenritshu_6']]
        self.ar_ze3 = np.round(np.array(ar_ze3, dtype='float16'), 0)

        ar_to1 = self.df[['tochi_shoritshu_1', 'tochi_shoritshu_2',
                          'tochi_shoritshu_3', 'tochi_shoritshu_4',
                          'tochi_shoritshu_5', 'tochi_shoritshu_6']]
        self.ar_to1 = np.round(np.array(ar_to1, dtype='float16'), 0)

        ar_to2 = self.df[['tochi_nirenritshu_1', 'tochi_nirenritshu_2',
                          'tochi_nirenritshu_3', 'tochi_nirenritshu_4',
                          'tochi_nirenritshu_5', 'tochi_nirenritshu_6']]
        self.ar_to2 = np.round(np.array(ar_to2, dtype='float16'), 0)

        ar_to3 = self.df[['tochi_sanrenritshu_1', 'tochi_sanrenritshu_2',
                          'tochi_sanrenritshu_3', 'tochi_sanrenritshu_4',
                          'tochi_sanrenritshu_5', 'tochi_sanrenritshu_6']]
        self.ar_to3 = np.round(np.array(ar_to3, dtype='float16'), 0)

        ar_incourse = self.df[['entry_course_1_no', 'entry_course_2_no',
                               'entry_course_3_no', 'entry_course_4_no',
                               'entry_course_5_no', 'entry_course_6_no']]
        self.ar_incourse = np.array(ar_incourse, dtype='int16')

        tenji_time = self.df[['tenji_time_1',
                              'tenji_time_2',
                              'tenji_time_3',
                              'tenji_time_4',
                              'tenji_time_5',
                              'tenji_time_6']]

        tenji_start_time = self.df[['tenji_start_time_1',
                                    'tenji_start_time_2',
                                    'tenji_start_time_3',
                                    'tenji_start_time_4',
                                    'tenji_start_time_5',
                                    'tenji_start_time_6']]

        self.tenji_time = np.array(tenji_time, dtype='float32')
        self.tenji_start_time = np.array(tenji_start_time, dtype='float32')

        ar_condition = self.df[['weather',
                                'temperature',
                                'water_temperature',
                                'water_hight',
                                'wind_speed',
                                'wind_vector']]
        self.ar_condition = np.round(np.array(ar_condition,
                                              dtype='float16'), 1)

        c1 = (self.ar_num - self.ar_incourse[:, 0].reshape(-1, 1) == 0)*np.arange(1, 7)
        c2 = (self.ar_num - self.ar_incourse[:, 1].reshape(-1, 1) == 0)*np.arange(1, 7)
        c3 = (self.ar_num - self.ar_incourse[:, 2].reshape(-1, 1) == 0)*np.arange(1, 7)
        c4 = (self.ar_num - self.ar_incourse[:, 3].reshape(-1, 1) == 0)*np.arange(1, 7)
        c5 = (self.ar_num - self.ar_incourse[:, 4].reshape(-1, 1) == 0)*np.arange(1, 7)
        c6 = (self.ar_num - self.ar_incourse[:, 5].reshape(-1, 1) == 0)*np.arange(1, 7)

        c1 = np.sum(c1, axis=1).reshape(-1, 1)
        c2 = np.sum(c2, axis=1).reshape(-1, 1)
        c3 = np.sum(c3, axis=1).reshape(-1, 1)
        c4 = np.sum(c4, axis=1).reshape(-1, 1)
        c5 = np.sum(c5, axis=1).reshape(-1, 1)
        c6 = np.sum(c6, axis=1).reshape(-1, 1)

        self.ar_incourse_num = np.concatenate([c1, c2, c3, c4, c5, c6], axis=1) - 1

    def ret_sanren_odds(self):
        sanren_odds_lis = []
        for i in tqdm(range(len(self.ar_th))):
            num1 = self.ar_th[i][0]
            num2 = self.ar_th[i][1]
            num3 = self.ar_th[i][2]
            odds = self.sanren_odds.iloc[i]['sanren_tan_{}{}{}'.format(num1, num2, num3)]
            sanren_odds_lis.append(odds)

        return np.array(sanren_odds_lis)

    def ret_all_sanren_odds(self):
        sanren_tan_col = self.df.columns[79:79+120]
        sanren_df = self.df[sanren_tan_col]

        col_names = ['sanren_tan_{}'.format(i) for i in self.sanren_indx]
        selected_data = sanren_df[col_names]

        res = selected_data.values.astype('float32')
        
        return res

    def ret_sanren_onehot(self):
        sanren = ret_sanren()
        th123 = self.ar_th[:, 0]*100 + self.ar_th[:, 1]*10 + self.ar_th[:, 2]*1
        sanren_onehot = (sanren - th123.reshape(-1, 1) == 0)*1

        return sanren_onehot.astype('float32')

    def ret_niren_odds(self):
        niren_odds_lis = []
        for i in tqdm(range(len(self.ar_th))):
            num1 = self.ar_th[i][0]
            num2 = self.ar_th[i][1]
            odds = self.niren_odds.iloc[i]['niren_tan_{}{}'.format(num1, num2)]
            niren_odds_lis.append(odds)

        return np.array(niren_odds_lis)

    def ret_all_niren_odds(self):
        niren_tan_col = self.df.columns[79+120+20:79+120+20+30]
        niren_df = self.df[niren_tan_col]
        
        col_names = ['niren_tan_{}'.format(i) for i in self.niren_indx]
        selected_data = niren_df[col_names]
        
        res = selected_data.values.astype('float32')
        
        return res

    def ret_niren_onehot(self):
        niren = ret_niren()
        th12 = self.ar_th[:, 0]*10 + self.ar_th[:, 1]*1
        niren_onehot = (niren - th12.reshape(-1, 1) == 0)*1

        return niren_onehot.astype('float32')

    def model_weights_random_init(self, init_ratio=0.0001):
        """
        モデルの重みをランダムに初期化する関数
        """
        # モデルの重みを取得する
        weights = self.model.get_weights()

        # 重みをランダムに初期化する
        for i, weight in enumerate(weights):
            if len(weight.shape) == 2:
                # 重み行列の場合、init_ratioの割合でランダム初期化する
                rand_mask = np.random.binomial(1, init_ratio, size=weight.shape)
                rand_weights = np.random.randn(*weight.shape) * rand_mask
                weights[i] = weight * (1 - rand_mask) + rand_weights

        # モデルの重みをセットする
        self.model.set_weights(weights)

# %%
