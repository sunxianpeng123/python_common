# encoding: utf-8

"""
@author: sunxianpeng
@file: test.py
@time: 2021/7/6 10:28
"""
from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
import pandas as pd
class Main():
    def __init__(self):
        pass

    def read_presto_data_presto(self,sql):
        # Presto
        engine = create_engine('presto://113.107.166.14:28080/hive/parquet')
        # sql = "SELECT plat,sourcegname,date from hive.parquet.live_show_online WHERE date = '20210601'  AND plat = 1 AND sourcegname IN ('王者荣耀','英雄联盟','绝地求生') limit 10"
        df = pd.read_sql(sql, engine)
        # print(df.head(5))
        return df

    def read_presto_data_mysql(self,sql):
        # Presto
        engine = create_engine('presto://113.107.166.14:28080/mysql149/anchor_goods_info')  # host是服务器ip，port是端口，hive指的是Presto的catalog，my_schema是hive的schema。
        # sql = "SELECT platform_id,room_id,sales_number_add_sum,sales_price_add,cast ( statistics_date AS VARCHAR ) AS statistics_date " \
        #       "from mysql149.anchor_goods_info.anchor_sales_week_info " \
        #       "WHERE platform_id = 71 AND cast(statistics_date AS VARCHAR )='2021-07-05' limit 10 "
        df = pd.read_sql(sql, engine)  # 和一般pandas从数据库中读取数据无任何区别，分析师们应该非常熟悉了。
        # print(df.head(5))
        return df

    def read_presto_data_test(self,sql):
        # Presto
        engine = create_engine('presto://113.107.166.14:28080/')  # host是服务器ip，port是端口，hive指的是Presto的catalog，my_schema是hive的schema。
        df = pd.read_sql(sql, engine)  # 和一般pandas从数据库中读取数据无任何区别，分析师们应该非常熟悉了。
        return df


if __name__ == '__main__':
    m = Main()
    sql_mysql = "SELECT platform_id,room_id,sales_number_add_sum,sales_price_add,cast ( statistics_date AS VARCHAR ) AS statistics_date " \
          "from mysql149.anchor_goods_info.anchor_sales_week_info " \
          "WHERE platform_id = 71 AND cast(statistics_date AS VARCHAR )='2021-07-05' limit 10 "
    sql_hive = "SELECT plat,sourcegname,date from hive.parquet.live_show_online WHERE date = '20210601'  AND plat = 1 AND sourcegname IN ('王者荣耀','英雄联盟','绝地求生') limit 10"
    df_mysql = m.read_presto_data_test(sql_mysql)
    df_hive = m.read_presto_data_test(sql_hive)
    print(df_mysql.head(5))
    print(df_hive.head(5))




