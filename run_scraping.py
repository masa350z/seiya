# %%
import os
import sys
import time
import datetime
import pandas as pd
import scraping_module as sc


def checkDate(year, month, day):
    try:
        newDataStr="%04d/%02d/%02d"%(year,month,day)
        newDate=datetime.datetime.strptime(newDataStr,"%Y/%m/%d")
        return True
    except ValueError:
        return False


field_dic = {'桐生': '01', '戸田': '02', '江戸川': '03',
             '平和島': '04', '多摩川': '05', '浜名湖': '06',
             '蒲郡': '07', '常滑': '08', '津': '09',
             '三国': '10', 'びわこ': '11', '住之江': '12',
             '尼崎': '13', '鳴門': '14', '丸亀': '15',
             '児島': '16', '宮島': '17', '徳山': '18',
             '下関': '19', '若松': '20', '芦屋': '21',
             '福岡': '22', '唐津': '23', '大村': '24'}
# %%
args = sys.argv

year = int(args[1])
# %%
year = 2021
data_path = 'datas/boatdata_{}.csv'.format(year)
# %%
if os.path.exists(data_path):
    dataframe = pd.read_csv(data_path)
else:
    dataframe = pd.DataFrame({})

for m in range(12):
    for d in range(31):
        if checkDate(year, m+1, d+1):
            #try:
            inpdate = int('{}{}{}'.format(year,
                                            str(m+1).zfill(2),
                                            str(d+1).zfill(2)))
            for f in sc.get_fields(inpdate):
                field_num = int(field_dic[f])
                for r in range(12):
                    race_num = r + 1
                    unique_num = int(str(inpdate) +
                                        str(field_num).zfill(2) +
                                        str(race_num).zfill(2))
                    race_indx = [] if len(dataframe) == 0 else list(dataframe['race_num'])
                    if not (unique_num in race_indx):
                        racedata = sc.RaceData(inpdate, field_num, race_num)
                        race_df = racedata.ret_race_df()

                        dataframe = pd.concat([dataframe, race_df]).sort_index()
                        dataframe.to_csv(data_path, index=False)

                        print(unique_num)
            #except Exception as e:
                #print(e)
                #time.sleep(10)
# %%
