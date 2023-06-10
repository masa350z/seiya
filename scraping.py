# %%
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
# %%
df = pd.read_csv('datas/boatdata.csv')
# %%


# %%
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
    wether_num = int(str(wether_wind_div[1]).split('is-weather')[1].split('"')[0])
    wind_num = int(str(wether_wind_div[2]).split('is-wind')[1].split('"')[0])

    wind_water_div = wether_div.find_all('span')
    tempreture = float(wind_water_div[1].text.split('℃')[0])
    wind_speed = float(wind_water_div[4].text.split('m')[0])
    water_tempreture = float(wind_water_div[6].text.split('℃')[0])
    water_hight = float(wind_water_div[8].text.split('cm')[0])

    return wether_num, wind_num, tempreture, wind_speed, \
        water_tempreture, water_hight


def get_preinfo(inpdate, field_num, race_num):
    url = 'https://www.boatrace.jp/owpc/pc/race/beforeinfo'
    params = {'hd': inpdate,
              'jcd': str(field_num).zfill(2),
              'rno': race_num}
    soup = get_soup(url, params=params)

    cose_approach_list, start_time_list = get_cose_approach_starttime(soup)

    tenjitime_list = get_tenjitime(soup)
    wether_num, wind_num, tempreture, wind_speed, \
        water_tempreture, water_hight = get_field_condition(soup)

    return cose_approach_list, start_time_list, tenjitime_list, \
        wether_num, wind_num, tempreture, wind_speed, \
        water_tempreture, water_hight


def get_chakujun_shussohyo(inpdate, field_num, race_num):
    url = 'https://www.boatrace.jp/owpc/pc/race/raceresult'
    params = {'hd': inpdate,
              'jcd': str(field_num).zfill(2),
              'rno': race_num}
    soup = get_soup(url, params=params)

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
                shussohyo.append(int(table[2+4*j].find_all('span')[0].text))

    return chakujun, shussohyo


def get_grade_shouritsu_exinfo(inpdate, field_num, race_num):
    url = 'https://www.boatrace.jp/owpc/pc/race/racelist'
    params = {'hd': inpdate,
              'jcd': str(field_num).zfill(2),
              'rno': race_num}

    soup = get_soup(url, params=params)

    grade_dic = {'A1': 0, 'A2': 1,
                 'B1': 2, 'B2': 3}
    grade_list = []
    for i in range(6):
        grade = soup.find_all('div', class_='is-fs11')[i*2].find('span').text
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

        ex_boat_list.append(ex_boat_num)
        ex_cose_list.append(ex_cose_approach)
        ex_start_list.append(ex_start_time)
        ex_result_list.append(ex_result)

    return grade_list, zenkoku_shouritsu_list, touchi_shouritsu_list,\
        f_l_s_list, motor_list, boat_list,\
        ex_boat_list, ex_cose_list, ex_start_list, ex_result_list


def get_computer_prediction(inpdate, field_num, race_num):
    url = 'https://www.boatrace.jp/owpc/pc/race/pcexpect'
    params = {'hd': inpdate,
              'jcd': str(field_num).zfill(2),
              'rno': race_num}

    soup = get_soup(url, params=params)

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


def get_sanrentan_odds(inpdate, field_num, race_num):
    url = 'https://www.boatrace.jp/owpc/pc/race/odds3t'
    params = {'hd': inpdate,
              'jcd': str(field_num).zfill(2),
              'rno': race_num}

    sanrentan_odds = get_odds(url, params)

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


def get_sanrenpuku_odds(inpdate, field_num, race_num):
    url = 'https://www.boatrace.jp/owpc/pc/race/odds3f'
    params = {'hd': inpdate,
              'jcd': str(field_num).zfill(2),
              'rno': race_num}
    oddslist = get_odds(url, params)
    sanrenpuku_num = [123, 124, 125, 126,
                      134, 234, 135, 235, 136, 236,
                      145, 245, 345, 146, 246, 346,
                      156, 256, 356, 456]

    return oddslist, sanrenpuku_num


def get_nirentan_puku_odds(inpdate, field_num, race_num):
    params = {'hd': inpdate,
              'jcd': str(field_num).zfill(2),
              'rno': race_num}
    url = 'https://www.boatrace.jp/owpc/pc/race/odds2tf'

    oddslist = get_odds(url, params)
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

    return oddslist_nirentan, oddslist_nirenpuku, nirentan_num, nirenpuku_num


def get_kakurenpuku_odds(inpdate, field_num, race_num):
    params = {'hd': inpdate,
              'jcd': str(field_num).zfill(2),
              'rno': race_num}
    url = 'https://www.boatrace.jp/owpc/pc/race/oddsk'

    oddslist = get_odds(url, params)
    kakurenpuku_num = [12,
                       13, 23,
                       14, 24, 34,
                       15, 25, 35, 45,
                       16, 26, 36, 46, 56]

    return oddslist, kakurenpuku_num


def get_tansho_hukusho_odds(inpdate, field_num, race_num):
    params = {'hd': inpdate,
              'jcd': str(field_num).zfill(2),
              'rno': race_num}
    url = 'https://www.boatrace.jp/owpc/pc/race/oddstf'

    oddslist = get_odds(url, params)
    oddslist_tansho = oddslist[:6]
    oddslist_fukusho = oddslist[6:]

    return oddslist_tansho, oddslist_fukusho, np.arange(6)+1


# %%
get_fields(20230609)
# %%
inpdate, field_num, race_num = 20230609, 1, 6
params = {'hd': inpdate,
          'jcd': str(field_num).zfill(2),
          'rno': race_num}
print(get_preinfo(inpdate, field_num, race_num))
print(get_grade_shouritsu_exinfo(inpdate, field_num, race_num))
print(get_computer_prediction(inpdate, field_num, race_num))
print(get_sanrentan_odds(inpdate, field_num, race_num))
print(get_sanrenpuku_odds(inpdate, field_num, race_num))
print(get_nirentan_puku_odds(inpdate, field_num, race_num))
print(get_kakurenpuku_odds(inpdate, field_num, race_num))
print(get_tansho_hukusho_odds(inpdate, field_num, race_num))
# %%
field_dic = {'桐生': '01', '戸田': '02', '江戸川': '03',
             '平和島': '04', '多摩川': '05', '浜名湖': '06',
             '蒲郡': '07', '常滑': '08', '津': '09',
             '三国': '10', 'びわこ': '11', '住之江': '12',
             '尼崎': '13', '鳴門': '14', '丸亀': '15',
             '児島': '16', '宮島': '17', '徳山': '18',
             '下関': '19', '若松': '20', '芦屋': '21',
             '福岡': '22', '唐津': '23', '大村': '24'}
# %%
class RaceData:
    def __init__(self, inpdate, field_num, race_num):
        self.params = {'hd': inpdate,
                       'jcd': str(field_num).zfill(2),
                       'rno': race_num}

    def get_fields(self):
        url = 'https://www.boatrace.jp/owpc/pc/race/index'
        soup = get_soup(url, params={'hd': self.params['hd']})

        field_list = []
        for i in soup.find_all('td', class_='is-arrow1 is-fBold is-fs15'):
            field_list.append(i.find('img')['alt'])

        return field_list

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
                    shussohyo.append(int(table[2+4*j].find_all('span')[0].text))

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

            ex_boat_list.append(ex_boat_num)
            ex_cose_list.append(ex_cose_approach)
            ex_start_list.append(ex_start_time)
            ex_result_list.append(ex_result)

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

    def get_tansho_hukusho_odds(self):
        url = 'https://www.boatrace.jp/owpc/pc/race/oddstf'

        oddslist = get_odds(url, self.params)
        oddslist_tansho = oddslist[:6]
        oddslist_fukusho = oddslist[6:]

        return oddslist_tansho, oddslist_fukusho, np.arange(6)+1
