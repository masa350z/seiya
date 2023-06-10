# %%
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
import time


def get_soup(url, params=None, retry=3):
    try_ = 0
    while try_ < retry:
        try:
            response = requests.get(url, params=params)
            try_ = retry
        except Exception as e:
            print(e)
            time.sleep(10)

    response.encoding = response.apparent_encoding
    soup = BeautifulSoup(response.text, 'html.parser')

    return soup


def get_fields(inpdate):
    url = 'https://www.boatrace.jp/owpc/pc/race/index'
    params = {'hd': inpdate}
    soup = get_soup(url, params=params)

    field_list = []
    for i in soup.find_all('td', class_='is-arrow1 is-fBold is-fs15'):
        field_list.append(i.find('img')['alt'])

    return field_list


def get_cose_approach_starttime(soup):
    cose = soup.find('tbody', class_='is-p10-0')

    cose_approach_list = []
    for i in cose.find_all('span'):
        if 'table1_boatImage1Number' in str(i):
            cose_approach_list.append(int(i.text))

    start_time_div = cose.find_all('span', class_='table1_boatImage1Time')
    start_time_list = [i.text for i in start_time_div]

    indx = np.array([np.arange(6)[np.array(cose_approach_list) == i+1][0]
                    for i in range(6)])
    start_time_list = [start_time_list[i] for i in indx]
    start_time_list = [0 if 'F' in i else float(i) for i in start_time_list]

    return cose_approach_list, start_time_list


def get_tenjitime(soup):
    table = soup.find_all('div', class_='table1')[1].find('table')
    racers_div = table.find_all('tbody')

    tenjitime_list = [float(r.find_all('td')[4].text) for r in racers_div]

    return tenjitime_list


def get_field_condition(soup):
    wether_div = soup.find('div', class_='weather1_body')

    wether_wind_div = wether_div.find_all('p')
    wether_num = int(str(wether_wind_div[1])
                     .split('is-weather')[1].split('"')[0])
    wind_num = int(str(wether_wind_div[2])
                   .split('is-wind')[1].split('"')[0])

    wind_water_div = wether_div.find_all('span')
    tempreture = float(wind_water_div[1].text.split('℃')[0])
    wind_speed = float(wind_water_div[4].text.split('m')[0])
    water_tempreture = float(wind_water_div[6].text.split('℃')[0])
    water_hight = float(wind_water_div[8].text.split('cm')[0])

    return wether_num, wind_num, tempreture, wind_speed, \
        water_tempreture, water_hight


def get_odds(url, params):
    soup = get_soup(url, params=params)
    oddslist = []
    odds = soup.find_all('td', class_='oddsPoint')
    if len(odds) == 0:
        return []
    else:
        for o in odds:
            if o.text == '欠場':
                return []
            else:
                oddslist.append(float(o.text.split('-')[0]))

        return oddslist


def ret_dataframe(input_list, columns, dtype):
    return pd.DataFrame([input_list],
                        columns=columns).astype(dtype)


def ret_preinfo_df(preinfo):
    cose_approach_list, start_time_list, tenjitime_list,\
        wether_num, wind_num, tempreture, wind_speed,\
        water_tempreture, water_hight = preinfo

    cose_approach_col = ['cose_1', 'cose_2', 'cose_3',
                         'cose_4', 'cose_5', 'cose_6']
    cose_approach_df = ret_dataframe(cose_approach_list,
                                     cose_approach_col, 'int16')

    start_time_col = ['start_time_1', 'start_time_2', 'start_time_3',
                      'start_time_4', 'start_time_5', 'start_time_6']
    start_time_df = ret_dataframe(start_time_list,
                                  start_time_col, 'float32')

    tenji_time_col = ['tenji_time_1', 'tenji_time_2', 'tenji_time_3',
                      'tenji_time_4', 'tenji_time_5', 'tenji_time_6']
    tenji_time_df = ret_dataframe(tenjitime_list,
                                  tenji_time_col, 'float32')

    input_list = [wether_num, wind_num, tempreture, wind_speed,
                  water_tempreture, water_hight]
    field_condition_col = ['wether_num', 'wind_num', 'tempreture',
                           'wind_speed', 'water_tempreture', 'water_hight']
    field_condition_df = ret_dataframe(input_list,
                                       field_condition_col, 'float32')

    dfs = [cose_approach_df, start_time_df, tenji_time_df, field_condition_df]

    return pd.concat(dfs, axis=1)


def ret_grade_shouritsu_exinfo_df(grade_shouritsu_exinfo):
    grade_list, zenkoku_shouritsu_list, touchi_shouritsu_list,\
        f_l_s_list, motor_list, boat_list,\
        ex_boat_list, ex_cose_list,\
        ex_start_list, ex_result_list = grade_shouritsu_exinfo

    grade_col = ['grade_1', 'grade_2', 'grade_3',
                 'grade_4', 'grade_5', 'grade_6']

    grade_df = ret_dataframe(grade_list, grade_col, 'int16')

    zenkoku_shouritsu_col = ['zenkoku_shouritsu_1', 'zenkoku_nirenritsu_1',
                             'zenkoku_sanrenritsu_1',
                             'zenkoku_shouritsu_2', 'zenkoku_nirenritsu_2',
                             'zenkoku_sanrenritsu_2',
                             'zenkoku_shouritsu_3', 'zenkoku_nirenritsu_3',
                             'zenkoku_sanrenritsu_3',
                             'zenkoku_shouritsu_4', 'zenkoku_nirenritsu_4',
                             'zenkoku_sanrenritsu_4',
                             'zenkoku_shouritsu_5', 'zenkoku_nirenritsu_5',
                             'zenkoku_sanrenritsu_5',
                             'zenkoku_shouritsu_6', 'zenkoku_nirenritsu_6',
                             'zenkoku_sanrenritsu_6']

    zenkoku_shouritsu_df = ret_dataframe(np.array(zenkoku_shouritsu_list)
                                         .reshape(-1),
                                         zenkoku_shouritsu_col, 'float32')

    touchi_shouritsu_col = ['touchi_shouritsu_1', 'touchi_nirenritsu_1',
                            'touchi_sanrenritsu_1',
                            'touchi_shouritsu_2', 'touchi_nirenritsu_2',
                            'touchi_sanrenritsu_2',
                            'touchi_shouritsu_3', 'touchi_nirenritsu_3',
                            'touchi_sanrenritsu_3',
                            'touchi_shouritsu_4', 'touchi_nirenritsu_4',
                            'touchi_sanrenritsu_4',
                            'touchi_shouritsu_5', 'touchi_nirenritsu_5',
                            'touchi_sanrenritsu_5',
                            'touchi_shouritsu_6', 'touchi_nirenritsu_6',
                            'touchi_sanrenritsu_6']

    touchi_shouritsu_df = ret_dataframe(np.array(touchi_shouritsu_list)
                                        .reshape(-1),
                                        touchi_shouritsu_col, 'float32')

    f_l_s_col = ['flying_count_1', 'latestart_count_1', 'average_start_time_1',
                 'flying_count_2', 'latestart_count_2', 'average_start_time_2',
                 'flying_count_3', 'latestart_count_3', 'average_start_time_3',
                 'flying_count_4', 'latestart_count_4', 'average_start_time_4',
                 'flying_count_5', 'latestart_count_5', 'average_start_time_5',
                 'flying_count_6', 'latestart_count_6', 'average_start_time_6']

    f_l_s_df = ret_dataframe(np.array(f_l_s_list)
                             .reshape(-1),
                             f_l_s_col, 'float32')

    motor_col = ['motor_no_1', 'motor_nirenritsu_1', 'motor_sanrenritsu_1',
                 'motor_no_2', 'motor_nirenritsu_2', 'motor_sanrenritsu_2',
                 'motor_no_3', 'motor_nirenritsu_3', 'motor_sanrenritsu_3',
                 'motor_no_4', 'motor_nirenritsu_4', 'motor_sanrenritsu_4',
                 'motor_no_5', 'motor_nirenritsu_5', 'motor_sanrenritsu_5',
                 'motor_no_6', 'motor_nirenritsu_6', 'motor_sanrenritsu_6']

    motor_df = ret_dataframe(np.array(motor_list)
                             .reshape(-1),
                             motor_col, 'float32')

    boat_col = ['boat_no_1', 'boat_nirenritsu_1', 'boat_sanrenritsu_1',
                'boat_no_2', 'boat_nirenritsu_2', 'boat_sanrenritsu_2',
                'boat_no_3', 'boat_nirenritsu_3', 'boat_sanrenritsu_3',
                'boat_no_4', 'boat_nirenritsu_4', 'boat_sanrenritsu_4',
                'boat_no_5', 'boat_nirenritsu_5', 'boat_sanrenritsu_5',
                'boat_no_6', 'boat_nirenritsu_6', 'boat_sanrenritsu_6']

    boat_df = ret_dataframe(np.array(boat_list)
                            .reshape(-1), boat_col,
                            'float32')

    ex_boatno_col = []
    for i in range(6):
        for j in range(12):
            ex_boatno_col.append('ex_boat_no_{}_{}'.format(i+1, j+1))

    ex_boatno_df = ret_dataframe(np.array(ex_boat_list)
                                 .reshape(-1),
                                 ex_boatno_col, 'int16')

    ex_cose_col = []
    for i in range(6):
        for j in range(12):
            ex_cose_col.append('ex_cose_{}_{}'.format(i+1, j+1))

    ex_cose_df = ret_dataframe(np.array(ex_cose_list)
                               .reshape(-1),
                               ex_cose_col, 'int16')

    ex_start_col = []
    for i in range(6):
        for j in range(12):
            ex_start_col.append('ex_start_{}_{}'.format(i+1, j+1))

    ex_start_df = ret_dataframe(np.array(ex_start_list)
                                .reshape(-1),
                                ex_start_col, 'float32')

    ex_result_col = []
    for i in range(6):
        for j in range(12):
            ex_result_col.append('ex_result_{}_{}'.format(i+1, j+1))

    ex_result_df = ret_dataframe(np.array(ex_result_list)
                                 .reshape(-1),
                                 ex_result_col, 'int16')

    dfs = [grade_df, zenkoku_shouritsu_df, touchi_shouritsu_df,
           f_l_s_df, motor_df, boat_df, ex_boatno_df, ex_cose_df,
           ex_start_df, ex_result_df]

    return pd.concat(dfs, axis=1)


def ret_computer_prediction_df(computer_prediction):
    comp_pred, confidence, pred_mark_list = computer_prediction

    comp_pred_col = ['comp_pred_{}'.format(i+1) for i in range(len(comp_pred))]

    comp_pred_df = ret_dataframe(comp_pred, comp_pred_col, 'int16')

    confidence_df = ret_dataframe([confidence], ['comfidence'], 'int16')

    pred_mark_col = ['comp_mark_{}'.format(i+1) for i in range(6)]

    pred_mark_df = ret_dataframe(pred_mark_list, pred_mark_col, 'int16')

    dfs = [comp_pred_df, confidence_df, pred_mark_df]

    return pd.concat(dfs, axis=1)


def ret_odds_df(sanrentan_odds, sanrentan_num,
                sanrenpuku_odds, sanrenpuku_num,
                nirentan_odds, nirenpuku_odds,
                nirentan_num, nirenpuku_num,
                kakurenpuku_odds, kakurenpuku_num,
                tansho_odds, fukusho_odds):

    sanrentan_col = ['sanrentan_{}'.format(i) for i in sanrentan_num]
    sanrentan_df = ret_dataframe(sanrentan_odds,
                                 sanrentan_col, 'float32')

    sanrenpuku_col = ['sanrenpuku_{}'.format(i) for i in sanrenpuku_num]
    sanrenpuku_df = ret_dataframe(sanrenpuku_odds,
                                  sanrenpuku_col, 'float32')

    nirentan_col = ['nirentan_{}'.format(i) for i in nirentan_num]
    nirentan_df = ret_dataframe(nirentan_odds,
                                nirentan_col, 'float32')

    nirenpuku_col = ['nirenpuku_{}'.format(i) for i in nirenpuku_num]
    nirenpuku_df = ret_dataframe(nirenpuku_odds,
                                 nirenpuku_col, 'float32')

    kakurenpuku_col = ['kakurenpuku_{}'.format(i) for i in kakurenpuku_num]
    kakurenpuku_df = ret_dataframe(kakurenpuku_odds,
                                   kakurenpuku_col, 'float32')

    tansho_col = ['tansho_{}'.format(i+1) for i in range(6)]
    tansho_df = ret_dataframe(tansho_odds,
                              tansho_col, 'float32')

    fukusho_col = ['fukusho_{}'.format(i+1) for i in range(6)]
    fukusho_df = ret_dataframe(fukusho_odds,
                               fukusho_col, 'float32')

    dfs = [sanrentan_df, sanrenpuku_df, nirentan_df, nirenpuku_df,
           kakurenpuku_df, tansho_df, fukusho_df]

    return pd.concat(dfs, axis=1)


class RaceData:
    def __init__(self, inpdate, field_num, race_num):
        self.params = {'hd': inpdate,
                       'jcd': str(field_num).zfill(2),
                       'rno': race_num}

    def get_preinfo(self):
        url = 'https://www.boatrace.jp/owpc/pc/race/beforeinfo'
        soup = get_soup(url, params=self.params)

        cose_approach_list, start_time_list = get_cose_approach_starttime(soup)

        tenjitime_list = get_tenjitime(soup)
        wether_num, wind_num, tempreture, wind_speed, \
            water_tempreture, water_hight = get_field_condition(soup)

        return cose_approach_list, start_time_list, tenjitime_list, \
            wether_num, wind_num, tempreture, wind_speed, \
            water_tempreture, water_hight

    def get_chakujun_shussohyo(self):
        url = 'https://www.boatrace.jp/owpc/pc/race/raceresult'
        soup = get_soup(url, params=self.params)

        race_result = soup.find('table', class_='is-w495')
        table = race_result.find_all('td')
        chakujun = []
        for i in range(6):
            if table[1+4*i].text in ['1', '2', '3', '4', '5', '6']:
                chakujun.append(int(table[1+4*i].text))

        shussohyo = []
        for i in ['1', '2', '3', '4', '5', '6']:
            for j in range(6):
                if table[1+4*j].text == i:
                    shussohyo.append(int(table[2+4*j]
                                         .find_all('span')[0].text))

        return chakujun, shussohyo

    def get_grade_shouritsu_exinfo(self):
        url = 'https://www.boatrace.jp/owpc/pc/race/racelist'
        soup = get_soup(url, params=self.params)

        grade_dic = {'A1': 0, 'A2': 1,
                     'B1': 2, 'B2': 3}
        grade_list = []
        for i in range(6):
            grade = soup.find_all('div',
                                  class_='is-fs11')[i*2].find('span').text
            grade_list.append(grade_dic[grade])

        shouritsu_div = soup.find_all('td', class_='is-lineH2')
        racers_tbody = soup.find_all('tbody', class_='is-fs12')

        zenkoku_shouritsu_list = []
        touchi_shouritsu_list = []

        f_l_s_list = []
        motor_list = []
        boat_list = []

        ex_boat_list = []
        ex_cose_list = []
        ex_start_list = []
        ex_result_list = []

        for i in range(6):
            zenkoku_shouritsu123 = shouritsu_div[1+5*i].text.split('\n')
            zenkoku_shouritsu123 = [float(z.replace(' ', '').replace('\r', ''))
                                    for z in zenkoku_shouritsu123[:3]]

            touchi_shouritsu123 = shouritsu_div[2+5*i].text.split('\n')
            touchi_shouritsu123 = [float(z.replace(' ', '').replace('\r', ''))
                                   for z in touchi_shouritsu123[:3]]

            zenkoku_shouritsu_list.append(zenkoku_shouritsu123)
            touchi_shouritsu_list.append(touchi_shouritsu123)

            f_l_s = shouritsu_div[0+5*i].text.split('\n')
            f_l_s = [float(z.replace(' ', '').replace('\r', '')
                           .replace('F', '').replace('L', ''))
                     for z in f_l_s[:3]]

            motor123 = shouritsu_div[3+5*i].text.split('\n')
            motor123 = [float(z.replace(' ', '').replace('\r', ''))
                        for z in motor123[:3]]

            boat123 = shouritsu_div[4+5*i].text.split('\n')
            boat123 = [float(z.replace(' ', '').replace('\r', ''))
                       for z in boat123[:3]]

            f_l_s_list.append(f_l_s)
            motor_list.append(motor123)
            boat_list.append(boat123)

            ex_info_div = racers_tbody[i].find_all('tr')

            ex_boat_num = [z['class']
                           for z in ex_info_div[0].find_all('td')[9:9+12]]
            ex_boat_num = [0 if len(z) == 0 else int(z[0][-1:])
                           for z in ex_boat_num]

            ex_cose_approach = [int(z.text.replace('\xa0', '0'))
                                for z in ex_info_div[1].find_all('td')]

            ex_start_time = [float(z.text.replace('\xa0', '0'))
                             for z in ex_info_div[2].find_all('td')]

            ex_result = [0 if z.text == '' else int(z.text)
                         for z in ex_info_div[3].find_all('td')]

            ex_boat_list.append(ex_boat_num[:12])
            ex_cose_list.append(ex_cose_approach[:12])
            ex_start_list.append(ex_start_time[:12])
            ex_result_list.append(ex_result[:12])

        return grade_list, zenkoku_shouritsu_list, touchi_shouritsu_list,\
            f_l_s_list, motor_list, boat_list,\
            ex_boat_list, ex_cose_list, ex_start_list, ex_result_list

    def get_computer_prediction(self):
        url = 'https://www.boatrace.jp/owpc/pc/race/pcexpect'
        soup = get_soup(url, params=self.params)

        comp_pred_div = soup.find('div', class_='numberSet2 h-clear')
        comp_pred = [int(i.text) for i in comp_pred_div.find_all('span')]
        confidence = int(soup.find('p', class_='state2_lv')['class'][1][-1:])

        pred_mark_dic = {'/static_extra/pc/images/icon_mark1_1.png': 1,  # ◎
                         '/static_extra/pc/images/icon_mark1_2.png': 2,  # ◯
                         '/static_extra/pc/images/icon_mark1_4.png': 3,  # △
                         '/static_extra/pc/images/icon_mark1_3.png': 4   # ✕
                         }

        racers_tbody = soup.find_all('tbody', class_='is-fs12')

        pred_mark_list = []
        for i in range(6):
            pred_mark = racers_tbody[i].find_all('td')[0].find('img')
            if pred_mark:
                pred_mark_list.append(pred_mark_dic[pred_mark['src']])
            else:
                pred_mark_list.append(0)

        return comp_pred, confidence, pred_mark_list

    def get_sanrentan_odds(self):
        url = 'https://www.boatrace.jp/owpc/pc/race/odds3t'
        sanrentan_odds = get_odds(url, self.params)

        sanrentan_num = []
        for i in range(6):
            for j in range(6):
                if not i == j:
                    for k in range(6):
                        if not (j == k or i == k):
                            sanrentan_num.append((i + 1)*100 +
                                                 (j + 1)*10 +
                                                 (k + 1)*1)

        sanrentan_num = np.array(sanrentan_num)
        sanrentan_num = sanrentan_num.reshape(6, 20).T
        sanrentan_num = list(sanrentan_num.reshape(120,))

        return sanrentan_odds, sanrentan_num

    def get_sanrenpuku_odds(self):
        url = 'https://www.boatrace.jp/owpc/pc/race/odds3f'
        oddslist = get_odds(url, self.params)
        sanrenpuku_num = [123, 124, 125, 126,
                          134, 234, 135, 235, 136, 236,
                          145, 245, 345, 146, 246, 346,
                          156, 256, 356, 456]

        return oddslist, sanrenpuku_num

    def get_nirentan_puku_odds(self):
        url = 'https://www.boatrace.jp/owpc/pc/race/odds2tf'

        oddslist = get_odds(url, self.params)
        oddslist_nirentan = oddslist[:30]
        oddslist_nirenpuku = oddslist[30:]
        nirentan_num = [12, 21, 31, 41, 51, 61,
                        13, 23, 32, 42, 52, 62,
                        14, 24, 34, 43, 53, 63,
                        15, 25, 35, 45, 54, 64,
                        16, 26, 36, 46, 56, 65]

        nirenpuku_num = [12,
                         13, 23,
                         14, 24, 34,
                         15, 25, 35, 45,
                         16, 26, 36, 46, 56]

        return oddslist_nirentan, oddslist_nirenpuku,\
            nirentan_num, nirenpuku_num

    def get_kakurenpuku_odds(self):
        url = 'https://www.boatrace.jp/owpc/pc/race/oddsk'

        oddslist = get_odds(url, self.params)
        kakurenpuku_num = [12,
                           13, 23,
                           14, 24, 34,
                           15, 25, 35, 45,
                           16, 26, 36, 46, 56]

        return oddslist, kakurenpuku_num

    def get_tansho_fukusho_odds(self):
        url = 'https://www.boatrace.jp/owpc/pc/race/oddstf'

        oddslist = get_odds(url, self.params)
        oddslist_tansho = oddslist[:6]
        oddslist_fukusho = oddslist[6:]

        return oddslist_tansho, oddslist_fukusho

    def ret_race_df(self):
        preinfo = self.get_preinfo()
        grade_shouritsu_exinfo = self.get_grade_shouritsu_exinfo()
        computer_prediction = self.get_computer_prediction()
        sanrentan_odds, sanrentan_num = self.get_sanrentan_odds()
        sanrenpuku_odds, sanrenpuku_num = self.get_sanrenpuku_odds()
        nirentan_odds, nirenpuku_odds, \
            nirentan_num, nirenpuku_num = self.get_nirentan_puku_odds()
        kakurenpuku_odds, kakurenpuku_num = self.get_kakurenpuku_odds()
        tansho_odds, fukusho_odds = self.get_tansho_fukusho_odds()

        race_num_df = pd.DataFrame([self.params['hd']], columns=['race_num'])

        preinfo_df = \
            ret_preinfo_df(preinfo)

        grade_shouritsu_exinfo_df = \
            ret_grade_shouritsu_exinfo_df(grade_shouritsu_exinfo)

        computer_prediction_df = \
            ret_computer_prediction_df(computer_prediction)

        odds_df = \
            ret_odds_df(sanrentan_odds, sanrentan_num,
                        sanrenpuku_odds, sanrenpuku_num,
                        nirentan_odds, nirenpuku_odds,
                        nirentan_num, nirenpuku_num,
                        kakurenpuku_odds, kakurenpuku_num,
                        tansho_odds, fukusho_odds)

        race_df = pd.concat([race_num_df,
                            preinfo_df,
                            grade_shouritsu_exinfo_df,
                            computer_prediction_df,
                            odds_df], axis=1)

        return race_df


# %%
field_dic = {'桐生': '01', '戸田': '02', '江戸川': '03',
             '平和島': '04', '多摩川': '05', '浜名湖': '06',
             '蒲郡': '07', '常滑': '08', '津': '09',
             '三国': '10', 'びわこ': '11', '住之江': '12',
             '尼崎': '13', '鳴門': '14', '丸亀': '15',
             '児島': '16', '宮島': '17', '徳山': '18',
             '下関': '19', '若松': '20', '芦屋': '21',
             '福岡': '22', '唐津': '23', '大村': '24'}

df = pd.read_csv('datas/boatdata.csv')
# %%
get_fields(20230609)
# %%
inpdate, field_num, race_num = 20230609, 1, 6
racedata = RaceData(inpdate, field_num, race_num)
race_df = racedata.ret_race_df()
# %%
