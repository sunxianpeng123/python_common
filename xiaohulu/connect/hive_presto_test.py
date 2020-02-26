# encoding: utf-8

"""
@author: sunxianpeng
@file: presto_test.py
@time: 2019/10/9 17:08
"""
from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
import pandas as pd
from pyhive import hive
import prestodb
from impala.dbapi import connect

class Main():
    def __init__(self):
        pass

    def test_presto_connect(self):
        # Presto
        engine = create_engine('presto://113.107.166.14:28080/hive/parquet')  # host是服务器ip，port是端口，hive指的是Presto的catalog，my_schema是hive的schema。
        print(engine)
        df = pd.read_sql("select * from live_schedule_gift where plat=2 and date ='20191008' limit 2", engine)  # 和一般pandas从数据库中读取数据无任何区别，分析师们应该非常熟悉了。
        print(df)


    def test_presto_connect1(self):
        conn = prestodb.dbapi.connect(
            host='113.107.166.14',
            port=28080,
            user='root',
            catalog='hive',
            schema='parquet'
        )
        cur = conn.cursor()
        cur.execute("select * from live_schedule_gift where plat=2 and date ='20191008' limit 2")
        rows = cur.fetchall()
        print(type(rows))

    def read_mysql_data(self,sql):
        engine = create_engine("mysql+pymysql://xiaohulu:6DgajISh!@113.107.166.14:53306/short_video_platform")
        df = pd.read_sql_query(sql,engine)
        return df

    def test_pandas_merge(self):
        platform_id = "201"
        video_tag_id_sql_201 = "select  platform_id,room_id ,tag_id from  anchor_tag_ref where platform_id =" + platform_id + "  GROUP BY platform_id,room_id limit 2"
        tag_id_df = m.read_mysql_data(video_tag_id_sql_201)
        print(tag_id_df)
        res_df = pd.merge(tag_id_df, tag_id_df.head(1), on=["room_id"], how="left").dropna(axis=0, how='any')

        print(res_df)

    def test_hive_connect(self):
        conn = connect(host='113.107.166.14',port =10001)
        cursor = conn.cursor()
        cursor.execute("select * from live_schedule_gift where plat=2 and date ='20191008' limit 2")

        # print(df)

if __name__ == '__main__':
    m = Main()
    m.test_pandas_merge()



