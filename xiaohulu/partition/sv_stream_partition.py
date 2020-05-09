# encoding: utf-8

"""
@author: sunxianpeng
@file: hive_bi_partition_sql.py
@time: 2019/10/14 19:40
"""
import os

def add_partition(table_name,platform_id):
    resultArr=[]

    # sql = "ALTER TABLE parquet.table_name  ADD IF NOT EXISTS PARTITION(date="+str(date)+",plat=" + str(plat) + ") " \
    #          "LOCATION 'hdfs://master20:8020/user/spark/parquet/short_video/sv_anchor_basic_info/date="+str(date)+"/plat=" + str(plat) + "';"

    one ="ALTER TABLE parquet." + table_name + " ADD IF NOT EXISTS PARTITION(date="
    two = ",plat="+str(platform_id)

    if table_name == "live_anchor_basic_info":
        three=") LOCATION 'hdfs://master20:8020/user/spark/parquet/live_show/anchor_basic_info/date="
    if table_name == "live_fans_info":
        three=") LOCATION 'hdfs://master20:8020/user/spark/parquet/live_show/fans_info/date="
    if table_name == "live_fans_list_info":
        three=") LOCATION 'hdfs://master20:8020/user/spark/parquet/live_show/fans_list_info/date="
    if table_name == "live_goods_info":
        three=") LOCATION 'hdfs://master20:8020/user/spark/parquet/live_show/goods_info/date="
    if table_name == "live_purchase_info":
        three = ") LOCATION 'hdfs://master20:8020/user/spark/parquet/live_show/purchase_info/date="

    four = "/plat="+str(platform_id) +"';"

    file_path = u"F:\PythonProjects\python_common/xiaohulu\data/"+str(platform_id)+"_"+table_name+"_hive_partition_sql.txt"
    f = open(file_path,"w")

    import datetime
    # now = datetime.datetime.now()#获取当前系统时间
    time_run = "2020-03-11 00:00:00"
    now = datetime.datetime.strptime(time_run, "%Y-%m-%d %H:%M:%S")#指定某个时间
    print (now)
    i= 0
    while 1:
        date = now + datetime.timedelta(days= i)
        date = str(date).split(" ")[0].replace("-","")
        if  int(date) <= 20210311:
            s = one + str(date) + two + three + str(date) + four
            print(s)
            f.write(s)
            f.write("\n")
            # resultArr.append(s)
            i +=1
        else:
            break
    f.close()



if __name__ == '__main__':
    anchor_basic_table_name = 'live_anchor_basic_info'
    fans_info_table_name = 'live_fans_info'
    fans_list_table_name = 'live_fans_list_info'
    goods_info_table_name = 'live_goods_info'
    purchase_info_table_name = 'live_purchase_info'

    tables = [anchor_basic_table_name,fans_info_table_name,fans_list_table_name,goods_info_table_name,purchase_info_table_name]
    plats = [71]
    print(plats)
    # ALTER TABLE parquet.label_parquet ADD IF NOT EXISTS PARTITION(plat=201,date=20190527,group=999) LOCATION 'hdfs://master20:8020/tmp/repartition/comment/plat=201/date=20190527/group=999';
    for table_name in tables:
        for plat in plats:
            add_partition(table_name,plat)
