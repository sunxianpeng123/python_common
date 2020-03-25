# encoding: utf-8

"""
@author: sunxianpeng
@file: hive_bi_partition_sql.py
@time: 2019/10/14 19:40
"""
import os

class Main():
    def __init__(self):
        pass




if __name__ == '__main__':
    plats = "71"
    plats = [x for x in plats.split(" ")]
    print(plats)
    # ALTER TABLE parquet.label_parquet ADD IF NOT EXISTS PARTITION(plat=201,date=20190527,group=999) LOCATION 'hdfs://master20:8020/tmp/repartition/comment/plat=201/date=20190527/group=999';
    for plat in plats :
        path = r"F:\PythonProjects\python_common\xiaohulu\data\\"+str(plat)+".txt"
        print(path)
        # if not os.path.exists(path):


        f = open(path, "w")

        sql = "ALTER TABLE parquet.audience_statistic_info  ADD IF NOT EXISTS PARTITION(plat=" + str(plat) + ") " \
                 "LOCATION 'hdfs://master20:8020/user/spark/parquet/live_show/audience_statistic_info/plat=" + str(plat) + "';"
        print(sql)
        f.write(sql)
        f.write("\n")
