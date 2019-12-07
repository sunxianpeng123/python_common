# encoding: utf-8

"""
@author: sunxianpeng
@file: hive_bi_partition_sql.py
@time: 2019/10/14 19:40
"""


class Main():
    def __init__(self):
        pass




if __name__ == '__main__':
    group = 100

    months = [4, 5, 6, 7, 8]
    table_name = ["new_add_video_duration","new_add_video_count","total_play_time","classify_topn_anchor"]
    hdfs_path_name = ["bi","video_id_count","total_play_time","classify_topn_anchor"]

    for i in range(len(table_name)):
        tbn = table_name[i]
        hpn = hdfs_path_name[i]
        path = r"F:\PythonProjects\python_study\xiaohulu\data\\"+tbn+".txt"
        f = open(path, "w")
        for month in months:
            for g in range(group):
                sql = "  ALTER TABLE temporary."+tbn+" ADD IF NOT EXISTS PARTITION(month="+str(month)+",group="+str(g)+") " \
                        "LOCATION 'hdfs://master20:8020/tmp/short_video/"+hpn+"/month="+str(month)+"/group="+str(g)+"';"
                print(sql)
                f.write(sql)
                f.write("\n")

# "		CREATE EXTERNAL TABLE `total_play_time`(
# 		  `room_id` string,
# 		  `totalview` double,
# 		  `tag_id` int,
# 		  `tag_name` string
# 		 )
# 		PARTITIONED BY (
# 		  `month` int,
# 		  `group` int)
# 		  STORED AS PARQUET
# 		LOCATION
# 		  'hdfs://master20:8020/tmp/short_video/total_play_time';"
# ===================================================================================
# "		CREATE EXTERNAL TABLE `classify_topn_anchor`(
# 		  `room_id` string,
# 		  `fenshishu` int,
# 		  `totalbofangliang` int,
# 		  `ranknum` Long,
# 		  `tag_id` int,
# 		  `tag_name` string
# 		 )
# 		PARTITIONED BY (
# 		  `month` int,
# 		  `group` int)
# 		  STORED AS PARQUET
# 		LOCATION
# 		  'hdfs://master20:8020/tmp/short_video/classify_topn_anchor'; "