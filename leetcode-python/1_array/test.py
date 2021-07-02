# encoding: utf-8

"""
@author: sunxianpeng
@file: test.py
@time: 2021/5/31 15:59
"""

import requests
from bs4 import BeautifulSoup
import pandas  as pd
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
from datetime import date
import time
import prettytable as pt
import sys
import json
import requests
from prettytable import PrettyTable
import time
import hmac
import hashlib
import base64
from urllib import parse
import requests
# 获取头榜数据#
from apscheduler.schedulers.blocking import BlockingScheduler


def sendmessage():
    # 把timestamp+"\n"+密钥当做签名字符串，使用HmacSHA256算法计算签名，然后进行Base64 encode
    # 最后再把签名参数再进行urlEncode，得到最终的签名（需要使用UTF-8字符集）
    timestamp = round(time.time() * 1000)
    secret = "SEC914d0fa1a30f6ca5864b5714104352deeed6402dfdc63870ee839395519b8702"

    secret_enc = secret.encode('utf-8')
    # 把timestamp+"\n"+密钥当做签名字符串
    string_to_sign = '{}\n{}'.format(timestamp, secret)
    string_to_sign_enc = string_to_sign.encode('utf-8')
    # 使用HmacSHA256算法计算签名
    hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
    # 进行Base64 encode把签名参数再进行urlEncode
    sign = parse.quote(base64.b64encode(hmac_code))

    url1 = 'http://www.toubang.tv/live/platform.html?gid=0&dt=0'
    response = requests.get(url1)
    html = response.content
    soup = BeautifulSoup(html.decode('utf-8'), 'html.parser')
    result = soup.find('tbody').find_all('td')
    try:
        plat_name0 = result[1].a.text.split('\n')[4]
        plat_name1 = result[12].a.text.split('\n')[4]
        plat_name2 = result[23].a.text.split('\n')[4]
        plat_name3 = result[34].a.text.split('\n')[4]
        plat_name4 = result[45].a.text.split('\n')[4]
        plat_name5 = result[56].a.text.split('\n')[4]
        plat_name6 = result[67].a.text.split('\n')[4]
        plat_name7 = result[78].a.text.split('\n')[4]

    except IndexError as e:
        print(e)

    try:
        gift_values0 = float(result[3].text[:-1]) * 10000
        gift_values1 = float(result[14].text[:-1]) * 10000
        gift_values2 = float(result[25].text[:-1]) * 10000
        gift_values3 = float(result[36].text[:-1]) * 10000
        gift_values4 = float(result[47].text[:-1]) * 10000
        gift_values5 = float(result[58].text[:-1]) * 10000
        gift_values6 = float(result[69].text[:-1]) * 10000
        gift_values7 = float(result[80].text[:-1]) * 10000

    except IndexError as e:
        print(e)

    plat_gift = {plat_name0: gift_values0, plat_name1: gift_values1, plat_name2: gift_values2, plat_name3: gift_values3,
                 plat_name4: gift_values4, plat_name5: gift_values5, plat_name6: gift_values6, plat_name7: gift_values7}

    print(plat_gift)
    try:
        huya_toubang = plat_gift['虎牙']
        douyu_toubang = plat_gift['斗鱼']
        qie_toubang = plat_gift['企鹅电竞']
        wangyi_toubang = plat_gift['网易CC']
        Bilibili_toubang = plat_gift['Bilibili']
        YY_toubang = plat_gift['YY']
        kuaishou_toubang = plat_gift['快手']
        fanxingkugou_toubang = plat_gift['酷狗']
    except IndexError as e:
        print(e)
    # 获取小葫芦数据#
    # Presto
    today = date.today().strftime("'%Y%m%d'")
    # engine = create_engine('presto://113.107.166.14:28080/hive/parquet')
    engine = create_engine('presto://192.168.75.161:8080/hive/parquet')
    hive_sql = ''' SELECT plat as "平台",sum(cast(price as double)*count) as "小葫芦收入" FROM hive.parquet.live_schedule_gift WHERE date =''' + today + '''and plat in (57,2,1,9,26,12,15,59)  and gift_type ='1'  group by plat order by  plat'''
    print(hive_sql)
    data = pd.read_sql(hive_sql, engine)
    print(data)
    data.iloc[0, 0] = '虎牙'
    data.iloc[1, 0] = '斗鱼'
    data.iloc[2, 0] = '企鹅电竞'
    data.iloc[3, 0] = '网易CC'
    data.iloc[4, 0] = ' Bilibili'
    data.iloc[5, 0] = '繁星酷狗'
    data.iloc[6, 0] = 'YY'
    data.iloc[7, 0] = '快手'
    data.loc[0:0, '头榜收入'] = huya_toubang
    data.loc[0:0, '差额'] = (data.iloc[0:1, 1] - huya_toubang) / huya_toubang
    data.loc[1:1, '头榜收入'] = douyu_toubang
    data.loc[1:1, '差额'] = (data.iloc[1:, 1] - douyu_toubang) / douyu_toubang
    data.loc[2:2, '头榜收入'] = qie_toubang
    data.loc[2:2, '差额'] = (data.iloc[2:, 1] - qie_toubang) / qie_toubang
    data.loc[3:3, '头榜收入'] = wangyi_toubang
    data.loc[3:3, '差额'] = (data.iloc[3:, 1] - wangyi_toubang) / wangyi_toubang
    data.loc[4:4, '头榜收入'] = Bilibili_toubang
    data.loc[4:4, '差额'] = (data.iloc[4:, 1] - Bilibili_toubang) / Bilibili_toubang
    data.loc[5:5, '头榜收入'] = fanxingkugou_toubang
    data.loc[5:5, '差额'] = (data.iloc[5:, 1] - fanxingkugou_toubang) / fanxingkugou_toubang
    data.loc[6:6, '头榜收入'] = YY_toubang
    data.loc[6:6, '差额'] = (data.iloc[6:, 1] - YY_toubang) / YY_toubang
    data.loc[7:7, '头榜收入'] = kuaishou_toubang
    data.loc[7:7, '差额'] = (data.iloc[7:, 1] - kuaishou_toubang) / kuaishou_toubang
    data.iloc[:, 1:3] = data.iloc[:, 1:3].round(decimals=2)
    data['差额'] = data['差额'].apply(lambda x: format(x, '.2%'))

    # 制作图表#
    year = '年'
    month = '月'
    day = '日'
    hour = '时'
    minute = '分'
    t = time.localtime(time.time())
    ptb = pt.PrettyTable()
    ptb.field_names = ['平台', '小葫芦收入', '头榜收入', '差值']
    for j in range(len(data)):
        ptb.add_row(data.iloc[j])
    ptb.add_row(
        ['统计截止时间为：', str(t.tm_year) + year + str(t.tm_mon) + month + str(t.tm_mday) + day + str(t.tm_hour) + hour + str(
            t.tm_min) + minute, '', ''])
    print(ptb)

    url = 'https://oapi.dingtalk.com/robot/send?access_token=95fd062a732789f7dc86cf9d53b6f86a1ad6b22fc9d8b53e491aff927d09165f&timestamp=' + str(
        timestamp) + '&sign=' + str(sign)

    HEADERS = {
        "Content-Type": "application/json ;charset=utf-8 "
    }
    message = str(ptb)
    String_textMsg = { \
        "msgtype": "text",
        "text": {"content": message},
        "at": {
            "isAtAll": False  # 如果需要@所有人，这些写1
        }
    }
    String_textMsg = json.dumps(String_textMsg)
    res = requests.post(url, data=String_textMsg, headers=HEADERS)
    print(res.text)


sendmessage()
