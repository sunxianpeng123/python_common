# encoding: utf-8

"""
@author: sunxianpeng
@file: wu.py
@time: 2019/10/11 13:42
"""
import sqlalchemy
from sqlalchemy.engine import create_engine
import pandas as pd
from pyhive import hive
import time
import datetime
class Main():
    def read_presto_data(self,sql):
        # Presto
        engine = create_engine('presto://113.107.166.14:28080/hive/parquet')  # host是服务器ip，port是端口，hive指的是Presto的catalog，my_schema是hive的schema。
        df = pd.read_sql(sql, engine)  # 和一般pandas从数据库中读取数据无任何区别，分析师们应该非常熟悉了。
        return df
    def read_hive_data(self,sql):
        conn = hive.Connection(host="1.0.1.38", auth="CUSTOM", username='hive', password="pvXxHTsdqrt8", port=10000,
                               database='tapro_atg')
        df = pd.read_sql("select * from sales_data_leisure_view", conn)

    def read_mysql_data(self,sql):
        engine = create_engine("mysql+pymysql://xiaohulu:6DgajISh!@113.107.166.14:53306/short_video_platform")
        df = pd.read_sql_query(sql,engine)
        return df

    def print_now_time(self):
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(nowTime)

    def df_merge(self,left, right, merge_keys,join_type):
        res_df = pd.merge(left, right, on=merge_keys, how=join_type)
        return  res_df

    def save_df_to_excel(self,df, path):
        df.to_excel(path)

    def date_to_timestamp(self,date):
        # 字符串转日期
        date = datetime.datetime.strptime(date,'%Y-%m-%d')
        #
        timestamp = time.mktime(date.timetuple())
        timestamp = str(timestamp).split(".")[0]
        print(str(date) +"___" + timestamp)
        return timestamp

    def make_sql(self,platform_id,begin_date,end_date,begin_create_time,end_create_time,type):
        # print("==========hive sql=============")
        # 新增视频时长
        video_time_new_add_sql = "SELECT room_id,sum( cast( video_duration AS DOUBLE ) ) FROM( " \
                                 " SELECT DISTINCT room_id,video_id,video_duration FROM hive.parquet.live_show_video " \
                                 "WHERE plat = " + platform_id + " AND create_time >= " + begin_create_time + " AND create_time <  " + end_create_time + " AND date >= '" + begin_date + "' AND date < '" + end_date + "' ) " \
                                  "GROUP BY room_id"

        # 新增视频数#
        video_num_new_add_sql = "SELECT	room_id,count( distinct video_id ) FROM parquet.live_show_video " \
                                "WHERE plat = " + platform_id + "  AND create_time >= " + begin_create_time + " AND create_time < " + end_create_time + " and date >= '" + begin_date + "' and date < '" + end_date + "' " \
                                                                                                                                                                                                                      "GROUP BY room_id"
        # 总视频播放量#
        video_sum_play_num_sql = "SELECT room_id,sum( t.VIEW ) AS totalview FROM " \
                                 "(SELECT room_id,video_id,max( " + type + " )- min( " + type + " ) AS VIEW FROM hive.parquet.live_show_video " \
                                                                                                "WHERE plat = " + platform_id + " AND date >= '" + begin_date + "' 	AND date < '" + end_date + "' " \
                                                                                                                                                                                                  "GROUP BY room_id,video_id 	) t " \
                                                                                                                                                                                                  "GROUP BY t.room_id ORDER BY totalview DESC"
        # 各品类Top10UP主播放量占比（不要在浏览器hive里面跑）
        anchor_topn_play_num_percent_sql = "select *  from ( select * ,row_number () over ( PARTITION BY room_id ORDER BY fenshishu DESC ) as ranknum from (" \
                                           "SELECT  room_id as room_id1, max(mplatform_followers_count) as fenshishu " \
                                           "FROM hive.parquet.live_show_author " \
                                           "WHERE date >= '" + begin_date + "' AND date < '" + end_date + "' AND plat = " + platform_id + " " \
                                                                                                                                          "group by  room_id) t1 inner join (" \
                                                                                                                                          "select room_id,sum(bofangliang)  as totalbofangliang from (" \
                                                                                                                                          "SELECT room_id,video_id,max( video_view_count ) - min( video_view_count )  as bofangliang " \
                                                                                                                                          "FROM hive.parquet.live_show_video " \
                                                                                                                                          "WHERE date >= '" + begin_date + "' AND date < '" + end_date + "' AND plat = " + platform_id + "  " \
                                                                                                                                                                                                                                         "GROUP BY room_id, video_id ) t " \
                                                                                                                                                                                                                                         "group by room_id) t2 on t1.room_id1 = t2.room_id ) s " \
                                                                                                                                                                                                                                         "where ranknum<11"
        # print("==========out put path=============")
        video_time_new_add_path = u"F:\PythonProjects\python_study/xiaohulu/data/"
        video_num_new_add_path = u"F:\PythonProjects\python_study/xiaohulu/data/"
        # print("=========sql 和 path 字典=============")
        sql_path_map = {}
        sql_path_map[video_time_new_add_sql] = (video_time_new_add_path, "新增视频时长")
        sql_path_map[video_num_new_add_sql] = (video_num_new_add_path, "新增视频数")
        return sql_path_map

if __name__ == '__main__':
    m = Main()
    print("==========parameters =============")
    # 4 5 6 7 8
    months = [4]
    type = "video_digg_count"  # video_view_count
    platform_id = "201"
    video_tag_id_sql_201 = "select room_id ,tag_id from  anchor_tag_ref where platform_id =" + platform_id + "  GROUP BY platform_id,room_id"
    tag_id_df = m.read_mysql_data(video_tag_id_sql_201)
    print("============执行操作=============")
    for month in months:
        print("***************************第 "+str(month)+" 月份******************************")
        next_month = month + 1
        begin_date = "2019-0" + str(month) + "-01"
        end_date = "2019-0" + str(next_month) + "-01"
        begin_create_time = m.date_to_timestamp(begin_date)
        end_create_time = m.date_to_timestamp(end_date)

        sql_path_map = m.make_sql(platform_id,begin_date.replace("-",""),end_date.replace("-",""),begin_create_time,end_create_time,type)
        for k in sql_path_map.keys():
            v = sql_path_map[k]
            path = v[0] + "month_"+str(month)+"_"+v[1]+".xlsx"
            print(v[1]+"###################")
            print(path)
            print(k)
            merge_keys = ["room_id"]  # ["platform_id", "room_id"]
            how = "left"
            m.print_now_time()
            # presto_df = m.read_presto_data(k)
            # try:
            #     res_df = m.df_merge(presto_df, tag_id_df, merge_keys, how).dropna(axis=0, how='any')
            #     print(res_df.head(5))
            #     m.save_df_to_excel(res_df, path)
            #     m.print_now_time()
            # except Exception as e:
            #     print("第 "+str(month)+" 月份,"+v[1]+"失败****************************** ")


