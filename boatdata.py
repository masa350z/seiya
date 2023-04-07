# %%
from transformers import BertTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np
import os


def add_str(inp, str_):
    ret = []
    for i in inp:
        temp = [str_.format(str(j)) for j in i]
        ret.append(temp)
    ret = np.array(ret)

    return ret


def ret_sanren_dic():
    sanren_dic, count = {}, 0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                j1, j2, j3 = i == j, j == k, i == k
                if not (j1 or j2 or j3):
                    sanren_dic['{}{}{}'.format(i+1, j+1, k+1)] = count
                    count += 1

    return sanren_dic


def ret_niren_dic():
    niren_dic, count = {}, 0
    for i in range(6):
        for j in range(6):
            if not i == j:
                niren_dic['{}{}'.format(i+1, j+1)] = count
                count += 1

    return niren_dic


def split_data(inp, tr_rate=0.6, val_rate=0.2):
    train_len = int(len(inp)*tr_rate)
    valid_len = int(len(inp)*val_rate)

    train = inp[:train_len]
    valid = inp[train_len:train_len+valid_len]
    test = inp[train_len+valid_len:]

    return train, valid, test


class BoatDataset:
    def __init__(self, ret_grade=True, sorted=True):
        self.ret_grade, self.sorted = ret_grade, sorted

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.df = pd.read_csv('datas/boatdata.csv')
        self.read_boatcsv()

        indx = np.zeros(self.ar_num.shape)
        for i in range(6):
            for j in range(6):
                indx[:, j] += (self.ar_num[:, j] == self.ar_incourse[:, i])*i

        # 艇番順をコース順に並べ替えるためのインデックス
        self.indx = indx.astype('int16')

        self.sanren_dic = ret_sanren_dic()
        self.niren_dic = ret_niren_dic()

        # BERTに入力するデータ（レース環境＋レーサー）の単語配列
        self.tokenized_inputs = self.make_tokenized_inputs(0.2)

        sanren_tan_col = self.df.columns[79:79+120]
        # 正解の３連単オッズ
        self.sanren_odds = self.df[sanren_tan_col]

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
        self.ar_grade = np.array(ar_grade)

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

        self.ar_field = np.array([str(int(ba))[-4:-2] for ba in self.df['index']])

        ar_condition = self.df[['weather',
                                'temperature',
                                'water_temperature',
                                'water_hight',
                                'wind_speed',
                                'wind_vector']]
        self.ar_condition = np.round(np.array(ar_condition,
                                              dtype='float16'), 1)

    def ret_sorted(self, inp):
        """
        艇番順に並んでいるデータを
        侵入コース順に並び替える関数
        """
        ret = []
        if len(inp) == len(self.indx):
            for i in range(len(inp)):
                ret.append(inp[i][self.indx[i]])

            ret = np.array(ret)

            return ret
        else:
            raise ValueError("error!")

    def ret_field_conditions(self):
        """
        そのレースのレース場、コンディションを
        単語化して並べた配列を返す関数
        """
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

        field_conditions = np.concatenate([fe.reshape(-1, 1), cd],
                                          axis=1,
                                          dtype='str')

        return field_conditions

    def ret_racers_data(self):
        """
        そのレースのレーサー、グレードの情報を
        単語化して並べた配列を返す関数
        """
        if self.sorted:
            num = self.ret_sorted(add_str(self.ar_num, 'num_{}'))
            grade = self.ret_sorted(add_str(self.ar_grade, 'grade_{}'))
        else:
            num = add_str(self.ar_num, 'num_{}')
            grade = add_str(self.ar_grade, 'grade_{}')

        num1 = np.concatenate([num[:, 0].reshape(-1, 1),
                               grade[:, 0].reshape(-1, 1)], axis=1)

        num2 = np.concatenate([num[:, 1].reshape(-1, 1),
                               grade[:, 1].reshape(-1, 1)], axis=1)

        num3 = np.concatenate([num[:, 2].reshape(-1, 1),
                               grade[:, 2].reshape(-1, 1)], axis=1)

        num4 = np.concatenate([num[:, 3].reshape(-1, 1),
                               grade[:, 3].reshape(-1, 1)], axis=1)

        num5 = np.concatenate([num[:, 4].reshape(-1, 1),
                               grade[:, 4].reshape(-1, 1)], axis=1)

        num6 = np.concatenate([num[:, 5].reshape(-1, 1),
                               grade[:, 5].reshape(-1, 1)], axis=1)

        if self.ret_grade:
            racers_data = np.concatenate([num1, num2,
                                          num3, num4,
                                          num5, num6], axis=1)
        else:
            racers_data = np.concatenate([num1[:, :1], num2[:, :1],
                                          num3[:, :1], num4[:, :1],
                                          num5[:, :1], num6[:, :1]], axis=1)
        return racers_data

    def ret_saiyou_senshu(self, n, half_len):
        counts = pd.DataFrame(self.ar_num[:half_len].reshape(-1)).value_counts()
        counts = counts[:-int(len(counts)*n)]
        saiyou = np.array(list(counts.index)).reshape(-1)

        return np.array(['num_{}'.format(i) for i in saiyou], dtype='str')

    def make_tokenized_inputs(self, n):
        """
        レース環境とレーサーの単語配列を取得し
        トークナイザーでインデックス化して
        BERTに入力可能なデータにして返す関数
        """
        fname = 'datas/tokenized_inputs'
        if self.ret_grade:
            fname += '_grade'
        if self.sorted:
            fname += '_sorted'
        fname += '_{}.npy'.format(n)

        if os.path.exists(fname):
            ret = np.load(fname)
            ret[ret > 20000] -= 20000
            return ret
        else:
            field_data = self.ret_field_conditions()

            seps = np.zeros(len(field_data)).astype('str')
            seps[seps == '0.0'] = '[SEP]'
            seps = seps.reshape(-1, 1)

            racers_data = self.ret_racers_data()

            half_len = int(len(field_data)*0.6)

            fdf = pd.DataFrame(field_data[:half_len].reshape(-1))
            fdf = np.array(fdf[~fdf.duplicated()]).reshape(-1).astype('str')

            # rdf = pd.DataFrame(racers_data[:half_len].reshape(-1))
            # rdf = np.array(rdf[~rdf.duplicated()]).reshape(-1).astype('str')
            rdf = self.ret_saiyou_senshu(n, half_len)

            self.tokenizer.add_tokens(list(fdf), special_tokens=True)
            self.tokenizer.add_tokens(list(rdf), special_tokens=True)
            self.tokenizer.add_tokens(['grade_A1', 'grade_A2', 'grade_B1', 'grade_B2'],
                                      special_tokens=True)

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
            np.save(fname, ret)

            ret[ret > 20000] -= 20000

            return ret

    def ret_sorted_th(self):
        sort = self.ret_sorted(np.tile(
            np.array([1, 2, 3, 4, 5, 6]),
            (len(self.ar_num), 1)))

        th1 = sort.reshape(-1, 1)[(self.ar_th[:, 0]-1)]
        th2 = sort.reshape(-1, 1)[(self.ar_th[:, 1]-1)]
        th3 = sort.reshape(-1, 1)[(self.ar_th[:, 2]-1)]
        th4 = sort.reshape(-1, 1)[(self.ar_th[:, 3]-1)]
        th5 = sort.reshape(-1, 1)[(self.ar_th[:, 4]-1)]
        th6 = sort.reshape(-1, 1)[(self.ar_th[:, 5]-1)]

        course_th = np.concatenate([th1, th2, th3, th4, th5, th6], axis=1)

        return course_th

    def ret_sanren_odds(self):
        sanren_odds_lis = []
        for i in tqdm(range(len(self.ar_th))):
            num1 = self.ar_th[i][0]
            num2 = self.ar_th[i][1]
            num3 = self.ar_th[i][2]
            odds = self.sanren_odds.iloc[i]['sanren_tan_{}{}{}'.format(num1, num2, num3)]
            sanren_odds_lis.append(odds)

        return np.array(sanren_odds_lis)

    def ret_sorted_odds(self):
        if os.path.exists('datas/sorted_odds.npy'):
            return np.load('datas/sorted_odds.npy')
        else:
            sort = self.ret_sorted(np.tile(np.array([1, 2, 3, 4, 5, 6]),
                                           (len(self.ar_num), 1)))

            dics = []
            for i in tqdm(range(len(sort))):
                temp_dic = {}
                for j in range(6):
                    temp_dic[j+1] = sort[i][j]
                dics.append(temp_dic)

            sanren_lis = []
            for i in range(6):
                for j in range(6):
                    for k in range(6):
                        j1 = i == j
                        j2 = j == k
                        j3 = i == k
                        if not (j1 or j2 or j3):
                            sanren_lis.append([i+1, j+1, k+1])

            ret_odds = []
            for i in tqdm(range(len(self.sanren_odds))):
                temp_odds = []
                dic = dics[i]
                odds = self.sanren_odds.iloc[i]
                for j in range(120):
                    temp = sanren_lis[j]
                    temp_num = '{}{}{}'.format(dic[temp[0]], dic[temp[1]], dic[temp[2]])
                    temp_odds.append(odds['sanren_tan_{}'.format(temp_num)])

                ret_odds.append(temp_odds)

            ret_odds = np.array(ret_odds)
            np.save('datas/sorted_odds.npy', ret_odds)

            return ret_odds

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


if __name__ == '__main__':
    bt = BoatDataset(0.1)
    bt = BoatDataset(0.2)
    bt = BoatDataset(0.3)
    bt = BoatDataset(0.5)
