# encoding: utf-8

"""
@author: sunxianpeng
@file: s1.py
@time: 2020/2/26 12:04
"""

from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
import pandas as pd
from pyhive import hive
import prestodb
from impala.dbapi import connect


def presto_connect():
    # Presto
    engine = create_engine(
        'presto://113.107.166.14:28080/hive/dws')  # host是服务器ip，port是端口，hive指的是Presto的catalog，my_schema是hive的schema。
    print('engine = {}'.format(engine))
    df = pd.read_sql("select * from ls_room_gift_history_20191230 where platform_id=2 limit 2",
                     engine)  # 和一般pandas从数据库中读取数据无任何区别，分析师们应该非常熟悉了。
    print(df)
    print(df.count())


def presto_connect1():
    conn = prestodb.dbapi.connect(
        host='113.107.166.14',
        port=28080,
        user='root',
        catalog='hive',
        schema='dws'
    )
    cur = conn.cursor()
    cur.execute("select * from ls_room_gift_history_20191230 where platform_id=2 limit 2")
    rows = cur.fetchall()
    print(type(rows))


if __name__ == '__main__':
    df = presto_connect()