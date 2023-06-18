# %%
from tqdm import tqdm
import pandas as pd
import numpy as np


def split_data(inp, tr_rate=0.6, val_rate=0.2):
    """
    データをトレーニングデータ、バリデーションデータ、テストデータに分割する関数

    Args:
        inp (array-like): 入力データ
        tr_rate (float): トレーニングデータの割合（デフォルト値: 0.6)
        val_rate (float): バリデーションデータの割合（デフォルト値: 0.2)

    Returns:
        tuple: トレーニングデータ、バリデーションデータ、テストデータのタプル
    """
    train_len = int(len(inp)*tr_rate)
    valid_len = int(len(inp)*val_rate)

    train = inp[:train_len]
    valid = inp[train_len:train_len+valid_len]
    test = inp[train_len+valid_len:]

    return train, valid, test


def ret_sanren():
    """
    3連単の番号リストを返す関数

    Returns:
        ndarray: 3連単の番号リスト
    """
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
    """
    2連単の番号リストを返す関数

    Returns:
        ndarray: 2連単の番号リスト
    """
    niren = []
    for i in range(6):
        for j in range(6):
            if not i == j:
                niren.append((i+1)*10 + (j+1)*1)

    return np.array(niren)


class BoatData:
    def __init__(self, race_field=None):
        self.df = pd.read_csv('datas/boatdata.csv')
        self.ar_field, self.df = self.ret_field(race_field)

    def ret_field(self, race_field):
        ar_field = np.array([str(int(ba))[-4:-2]
                            for ba in self.df['race_num']],
                            dtype='int16')
        if race_field:
            df = self.df[ar_field == race_field].reset_index(drop=True)
            ar_field = np.array([str(int(ba))[-4:-2]
                                for ba in df['race_num']],
                                dtype='int16')

        return ar_field, df

    def ret_entryno_th_grade(self):
        entry_no = self.df[['entry_no_1', 'entry_no_2', 'entry_no_3',
                            'entry_no_4', 'entry_no_5', 'entry_no_6']]
        entry_no = np.array(entry_no, dtype='int16')

        th = self.df[['th_1', 'th_2', 'th_3',
                      'th_4', 'th_5', 'th_6']]
        th = np.array(th, dtype='int16')

        grade = self.df[['grade_1', 'grade_2', 'grade_3',
                         'grade_4', 'grade_5', 'grade_6']]
        grade = np.array(grade).astype('int16')

        return entry_no, th, grade

    def ret_shouritsu(self):
        zenkoku_touchi = []
        for shouritsu_range in ['zenkoku', 'touchi']:
            shouritsu = []
            for i in range(6):
                shouritsu_x = self.df[['{}_shouritsu_{}'.format(shouritsu_range, i+1),
                                       '{}_nirenritsu_{}'.format(shouritsu_range, i+1),
                                       '{}_sanrenritsu_{}'.format(shouritsu_range, i+1)]]
                shouritsu_x = np.array(shouritsu_x, dtype='float32')

                shouritsu.append(shouritsu_x)

            shouritsu = np.array(shouritsu)
            shouritsu = shouritsu.transpose(1, 0, 2)

            zenkoku_touchi.append(shouritsu)

        return zenkoku_touchi

    def ret_flying_latestart(self):
        flying_latestart = []
        for i in range(6):
            f_l = self.df[['flying_count_{}'.format(i+1),
                           'latestart_count_{}'.format(i+1)]]
            f_l = np.array(f_l, dtype='int16')

            flying_latestart.append(f_l)

        flying_latestart = np.array(flying_latestart)
        flying_latestart = flying_latestart.transpose(1, 0, 2)

        return flying_latestart

    def ret_average_starttime(self):
        average_start_time = []
        for i in range(6):
            a_s_t = self.df['average_start_time_{}'.format(i+1)]
            a_s_t = np.array(a_s_t, dtype='float32')

            average_start_time.append(a_s_t)

        average_start_time = np.array(average_start_time)
        average_start_time = average_start_time.transpose(1, 0)

        return average_start_time

    def ret_motor_boat_no(self):
        motor_boat_no = []
        for i in range(6):
            m_b_t = self.df[['motor_no_{}'.format(i+1),
                             'boat_no_{}'.format(i+1)]]
            m_b_t = np.array(m_b_t, dtype='int16')

            motor_boat_no.append(m_b_t)

        motor_boat_no = np.array(motor_boat_no)
        motor_boat_no = motor_boat_no.transpose(1, 0, 2)

        return motor_boat_no

    def ret_motor_boat_shouritsu(self):
        motor_boat = []
        for m_b in ['motor', 'boat']:
            motor_boat_shouritsu = []
            for i in range(6):
                m_b_s = self.df[['{}_nirenritsu_{}'.format(m_b, i+1),
                                 '{}_sanrenritsu_{}'.format(m_b, i+1)]]
                m_b_s = np.array(m_b_s, dtype='float32')

                motor_boat_shouritsu.append(m_b_s)

            motor_boat_shouritsu = np.array(motor_boat_shouritsu)
            motor_boat_shouritsu = motor_boat_shouritsu.transpose(1, 0, 2)

            motor_boat.append(motor_boat_shouritsu)

        return motor_boat

    def ret_ex_data(self):
        ex_data = []
        for i in range(6):
            temp = []
            for j in range(14):
                ex_no_cose_start_result = self.df[['ex_boat_no_{}_{}'.format(i+1, j+1),
                                                   'ex_cose_{}_{}'.format(i+1, j+1),
                                                   'ex_start_{}_{}'.format(i+1, j+1),
                                                   'ex_result_{}_{}'.format(i+1, j+1)]]
                ex_no_cose_start_result = np.array(ex_no_cose_start_result, dtype='float32')

                temp.append(ex_no_cose_start_result)
            ex_data.append(temp)

        ex_data = np.array(ex_data)
        ex_data = ex_data.transpose(2, 0, 3, 1)

        ex_no = ex_data[:, :, 0].astype('int16')
        ex_cose = ex_data[:, :, 1].astype('int16')
        ex_start = ex_data[:, :, 2]
        ex_result = ex_data[:, :, 3].astype('int16')

        return ex_no, ex_cose, ex_start, ex_result

    def ret_incose(self):
        in_cose = []
        for i in range(6):
            cose = self.df['cose_{}'.format(i+1)]
            cose = np.array(cose, dtype='int16')

            in_cose.append(cose)

        in_cose = np.array(in_cose)
        in_cose = in_cose.transpose(1, 0)

        return in_cose

    def ret_start_tenji(self):
        start_tenji = []
        for i in range(6):
            time = self.df[['start_time_{}'.format(i+1),
                            'tenji_time_{}'.format(i+1)]]
            time = np.array(time, dtype='float32')

            start_tenji.append(time)

        start_tenji = np.array(start_tenji)
        start_tenji = start_tenji.transpose(1, 0, 2)

        start_time = start_tenji[:, :, 0]
        tenji_time = start_tenji[:, :, 1]

        return start_time, tenji_time

    def ret_field_condition(self):
        conditions = self.df[['wether_num',
                              'wind_num',
                              'tempreture',
                              'wind_speed',
                              'water_tempreture',
                              'water_hight']]
        conditions = np.array(conditions, dtype='float32')

        wether = conditions[:, 0].astype('int16')
        wind = conditions[:, 1].astype('int16')
        tempreture = conditions[:, 2]
        wind_speed = conditions[:, 3]
        water_tempreture = conditions[:, 4]
        water_hight = conditions[:, 5]

        return wether, wind, tempreture, wind_speed, water_tempreture, water_hight

    def ret_computer_prediction(self):
        df_col = ['comp_pred_{}'.format(i+1) for i in range(26)]
        computer_prediction = self.df[df_col]
        computer_prediction = np.array(computer_prediction, dtype='int16')

        computer_confidence = self.df['comfidence']
        computer_confidence = np.array(computer_confidence, dtype='int16')

        df_col = ['comp_mark_{}'.format(i+1) for i in range(6)]
        prediction_mark = np.array(self.df[df_col], dtype='int16')

        return computer_prediction, computer_confidence, prediction_mark

    def ret_sanrentan_odds(self):
        df_col = ['sanrentan_{}'.format(i) for i in ret_sanren()]
        odds = self.df[df_col]
        odds = np.array(odds, dtype='float32')

        return np.where(odds == 0, 1, odds)

    def ret_nirentan_odds(self):
        df_col = ['nirentan_{}'.format(i) for i in ret_niren()]
        odds = self.df[df_col]
        odds = np.array(odds, dtype='float32')

        return np.where(odds == 0, 1, odds)


class BoatDataset(BoatData):
    def __init__(self, race_field=None):
        super().__init__(race_field)

        self.sanren_indx = ret_sanren()
        self.sanren_odds = self.ret_sanrentan_odds()

        self.niren_indx = ret_niren()
        self.niren_odds = self.ret_nirentan_odds()

        self.entry_no, self.th, self.grade = self.ret_entryno_th_grade()

        self.incose = self.ret_incose()

        self.zenkoku_shouritsu, self.touchi_shouritsu = self.ret_shouritsu()

        self.flying_latestart = self.ret_flying_latestart()

        self.average_starttime = self.ret_average_starttime()

        self.motor_boat_no = self.ret_motor_boat_no()

        self.motor_shouritsu, self.boat_shouritsu = self.ret_motor_boat_shouritsu()

        self.ex_no, self.ex_cose, self.ex_start, self.ex_result = self.ret_ex_data()

        self.start_time, self.tenji_time = self.ret_start_tenji()

        self.wether, self.wind, self.tempreture, self.wind_speed, self.water_tempreture, self.water_hight = self.ret_field_condition()

        self.computer_prediction, self.computer_confidence, self.prediction_mark = self.ret_computer_prediction()

        self.sanrentan_odds = self.ret_sanrentan_odds()

        self.nirentan_odds = self.ret_nirentan_odds()


# %%
