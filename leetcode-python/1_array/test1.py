# encoding: utf-8

"""
@author: sunxianpeng
@file: test1.py
@time: 2021/5/31 16:05
"""
import pandas as pd
from datetime import date
from sqlalchemy import create_engine


def read_presto_data(sql):
    # Presto
    engine = create_engine(
        'presto://113.107.166.14:28080/hive/parquet')  # host是服务器ip，port是端口，hive指的是Presto的catalog，my_schema是hive的schema。
    engine = create_engine('presto://113.107.166.14:28080/hive/parquet')
    df = pd.read_sql(sql, engine)  # 和一般pandas从数据库中读取数据无任何区别，分析师们应该非常熟悉了。
    return df

# today = date.today().strftime("'%Y%m%d'")
# # engine = create_engine('presto://113.107.166.14:28080/hive/parquet')
# engine = create_engine('presto://113.107.166.14:28080/hive/parquet')
# hive_sql = ''' SELECT plat as "平台",sum(cast(price as double)*count) as "小葫芦收入" FROM hive.parquet.live_schedule_gift WHERE date =''' + today + '''and plat in (57,2,1,9,26,12,15,59)  and gift_type ='1'  group by plat order by  plat'''
# print(hive_sql)
# data = pd.read_sql(hive_sql, engine)
if __name__ == '__main__':
    # engine = create_engine('presto://192.168.75.161:8080/hive/parquet')
    # sql = "select * from hive.parquet.live_goods_info where date='20210531' and plat=71 limit 10"
    # print(sql)
    # data = pd.read_sql(sql, engine)
    # print(data)

    today = date.today().strftime("'%Y%m%d'")
    # engine = create_engine('presto://113.107.166.14:28080/hive/parquet')
    engine = create_engine('presto://113.107.166.14:28080/hive/parquet')
    hive_sql = ''' SELECT plat as "平台",sum(cast(price as double)*count) as "小葫芦收入" FROM hive.parquet.live_schedule_gift WHERE date =''' + today + '''and plat in (57,2,1,9,26,12,15,59)  and gift_type ='1'  group by plat order by  plat'''
    print(hive_sql)
    data = pd.read_sql(hive_sql, engine)
    print(data)