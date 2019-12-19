# encoding: utf-8

"""
@author: sunxianpeng
@file: 1_indoor.py
@time: 2019/10/11 19:29
"""

import datetime
import time
class Main():
    def __init__(self):
        pass

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
        video_sum_play_num_sql =  "SELECT room_id,sum( t.VIEW ) AS totalview FROM " \
                                  "(SELECT room_id,video_id,max( " + type + " )- min( " + type + " ) AS VIEW FROM parquet.live_show_video " \
                                  "WHERE plat = " + platform_id + " AND date >= '" + begin_date + "' 	AND date < '" + end_date + "' " \
                                  "GROUP BY room_id,video_id 	) t " \
                                  "GROUP BY t.room_id ORDER BY totalview DESC"
        # 各品类Top10UP主播放量占比（不要在浏览器hive里面跑）
        anchor_topn_play_num_percent_sql =  "select *  from ( select * ,row_number () over ( PARTITION BY room_id ORDER BY fenshishu DESC ) as ranknum from (" \
                                            "SELECT  room_id as room_id1, max(mplatform_followers_count) as fenshishu " \
                                            "FROM parquet.live_show_author " \
                                            "WHERE date >= '" + begin_date + "' AND date < '" + end_date + "' AND plat = " + platform_id + " " \
                                            "group by  room_id) t1 inner join (" \
                                            "select room_id,sum(bofangliang)  as totalbofangliang from (" \
                                            "SELECT room_id,video_id,max( " + type + " ) - min( " + type + " )  as bofangliang " \
                                            "FROM parquet.live_show_video " \
                                            "WHERE date >= '" + begin_date + "' AND date < '" + end_date + "' AND plat = " + platform_id + "  " \
                                             "group by room_id) t2 on t1.room_id1 = t2.room_id ) s " \
                                             "where ranknum<11"
        # print("==========out put path=============")
        video_time_new_add_path = u"F:\PythonProjects\python_study/xiaohulu/data/"
        video_num_new_add_path = u"F:\PythonProjects\python_study/xiaohulu/data/"
        video_sum_play_num_path = u"F:\PythonProjects\python_study/xiaohulu/data/"
        anchor_topn_play_num_percent_path = u"F:\PythonProjects\python_study/xiaohulu/data/"
        # print("=========sql 和 path 字典=============")
        sql_path_map = {}
        sql_path_map[video_time_new_add_sql] = (video_time_new_add_path, "新增视频时长")
        sql_path_map[video_num_new_add_sql] = (video_num_new_add_path, "新增视频数")
        sql_path_map[video_sum_play_num_sql] = (video_sum_play_num_path, "总视频播放量")
        sql_path_map[anchor_topn_play_num_percent_sql] = (anchor_topn_play_num_percent_path, "各品类Top10UP主播放量占比")
        return sql_path_map

    def date_to_timestamp(self,date):
        # 字符串转日期
        date = datetime.datetime.strptime(date,'%Y-%m-%d')
        #
        timestamp = time.mktime(date.timetuple())
        timestamp = str(timestamp).split(".")[0]
        print(str(date) +"___" + timestamp)
        return timestamp
if __name__ == '__main__':
    m = Main()
    print("==========parameters =============")
    # 4 5 6 7 8
    # months = [4,5,6,7,8]
    months = [4]
    type = "video_digg_count"  # video_view_count
    platform_id = "201"

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
            print("month_"+str(month)+"_"+v[1])
            print(k)
