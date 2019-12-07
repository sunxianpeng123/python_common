# encoding: utf-8

"""
@author: sunxianpeng
@file: sql_test.py
@time: 2019/10/17 10:23
"""

import time
import datetime
class Main():
    def __init__(self):
        pass

    def make_sql(self,platform_id,begin_date,end_date,num=1001,begin_create_time="",end_create_time=""):
        # 新增视频数
        sql_0 = "SELECT  t.room_id,count( t.video_id ) AS video_id_count " \
                                 "FROM (SELECT room_id, video_id  " \
                                 "FROM parquet.live_show_video " \
                                 "WHERE plat = "+platform_id+" AND create_time >= "+begin_create_time+" AND create_time < "+end_create_time+" AND date >= '"+begin_date+"' AND date < '"+end_date+"' " \
                                 "GROUP BY room_id, video_id ) t " \
                                 "GROUP BY t.room_id"
        # 新增视频时长
        sql_00 = "SELECT t2.room_id,sum( t2.date_sum ) AS video_duration " \
        "FROM (SELECT t1.room_id,sum( cast( t1.video_duration AS DOUBLE ) ) AS date_sum " \
        "FROM ( SELECT room_id, video_id,max( video_duration ) AS video_duration,min( date ) AS date " \
        "FROM hive.parquet.live_show_video " \
        "WHERE plat = "+platform_id+"  AND create_time >= "+begin_create_time+" AND create_time < "+end_create_time+" AND date >= '"+begin_date+"' AND date < '"+end_date+"' " \
        "GROUP BY room_id, video_id  ) t1 " \
        "GROUP BY t1.room_id,t1.date ) t2  " \
        "GROUP BY room_id"""
        # 总视频播放量
        sql_1 = "SELECT r1.room_id_r1,r1.totalview,r2.tag_id,r2.tag_name  " \
                "FROM (SELECT  t2.room_id as room_id_r1,   sum( t2.VIEW ) AS  totalview " \
                "FROM   (  SELECT t1.room_id,  t1.video_id,  ( max( t1.max_video_digg_count ) - min( t1.min_video_digg_count ) ) AS VIEW " \
                "FROM  ( SELECT room_id, video_id, max( video_digg_count ) AS max_video_digg_count,  min( video_digg_count ) AS min_video_digg_count " \
                "FROM   parquet.live_show_video " \
                "WHERE plat = "+platform_id+"  AND date >= '"+begin_date+"' AND date < '"+end_date+"' GROUP BY room_id,  video_id,  date ) t1 " \
                "GROUP BY t1.room_id, t1.video_id  ) t2" \
                " GROUP BY  t2.room_id ) r1 " \
                "LEFT outer JOIN  " \
                "( SELECT room_id as room_id_r2, tag_id1 AS tag_id, tag_name " \
                "FROM(SELECT DISTINCT room_id, tag_id AS tag_id1 " \
                "FROM temporary.anchor_tag_ref " \
                "WHERE plat = "+platform_id+")  a " \
                "LEFT outer JOIN( SELECT  tag_id AS tag_id2, tag_name " \
                "FROM temporary.anchor_tag_config" \
                " WHERE plat = "+platform_id+" ) b ON a.tag_id1 = b.tag_id2 ) r2 on r1.room_id_r1 = r2.room_id_r2 " \
                "WHERE tag_id is not null"


        sql_2 = "select * from(" \
                "SELECT s1.room_id,s1.fenshishu,s1.totalbofangliang ,tag_id,tag_name,row_number ( ) over ( PARTITION BY tag_name ORDER BY fenshishu DESC ) as ranknum " \
                "FROM(SELECT t2.room_id,t1.fenshishu,t2.totalbofangliang " \
                "FROM(SELECT room_id AS room_id1,max( mplatform_followers_count ) AS fenshishu " \
                "FROM parquet.live_show_author " \
                "WHERE date >= '"+begin_date+"' AND date < '"+end_date+"' AND plat = "+platform_id+" GROUP BY room_id ) t1 " \
                "INNER JOIN (" \
                "SELECT room_id,sum( bofangliang ) AS totalbofangliang " \
                "FROM(SELECT room_id,video_id,max( video_digg_count ) - min( video_digg_count ) AS bofangliang " \
                "FROM live_show_video " \
                "WHERE date >= '"+begin_date+"' AND date < '"+end_date+"' AND plat = "+platform_id+" " \
                "GROUP BY room_id,video_id) t " \
                "GROUP BY room_id) t2 ON t1.room_id1 = t2.room_id ) s1 " \
                "left outer join(" \
                "SELECT room_id,tag_id1 AS tag_id,tag_name " \
                "FROM(SELECT DISTINCT room_id, tag_id AS tag_id1 FROM temporary.anchor_tag_ref " \
                "WHERE plat = "+platform_id+")  a " \
                "LEFT outer JOIN ( " \
                "SELECT  tag_id AS tag_id2, tag_name " \
                "FROM temporary.anchor_tag_config " \
                "WHERE plat = "+platform_id+" ) b ON a.tag_id1 = b.tag_id2 )  s2 on s1.room_id = s2.room_id " \
                "WHERE tag_id is not null ) s3 " \
                "where ranknum <"+str(num)+" and tag_id is not null"
        # 新增视频播放量
        sql_3 = "SELECT tag_id,tag_name,sum(view) as totalview " \
                "FROM(SELECT t2.room_id,t1.video_id,t1.view,tag_id,tag_name " \
                "from(SELECT room_id as room_id1,video_id,max(max_date) - min(min_date) as view " \
                "FROM (SELECT room_id,video_id,max( video_digg_count ) as max_date, min( video_digg_count ) as min_date " \
                "from parquet.live_show_video " \
                "WHERE plat =  "+platform_id+" AND create_time >= "+begin_create_time+" AND create_time <  "+end_create_time+" and  date >= '"+begin_date+"'  and  date <  '"+end_date+"' " \
                "GROUP BY room_id,video_id,date)" \
                "GROUP BY room_id,video_id) t1 " \
                "left outer JOIN(SELECT	room_id,tag_id1 AS tag_id,	tag_name " \
                "FROM(SELECT DISTINCT room_id, tag_id AS tag_id1 " \
                "FROM temporary.anchor_tag_ref " \
                "WHERE plat =  "+platform_id+")  a LEFT outer JOIN ( SELECT  tag_id AS tag_id2, tag_name " \
                "FROM temporary.anchor_tag_config " \
                "WHERE plat =  "+platform_id+" ) b ON a.tag_id1 = b.tag_id2 )t2 on t1.room_id1 = t2.room_id " \
                "WHERE tag_id is not null)" \
                "GROUP BY tag_id,tag_name"

        sql_4 = "SELECT tag_id,tag_name,sum(view) as totalview " \
                "FROM(SELECT t3.room_id,t2.video_id,t2.view,tag_id,tag_name " \
                "from(SELECT t1.room_id as room_id1,t1.video_id,max(t1.max_date) - min(t1.min_date) as view " \
                "FROM (SELECT room_id,video_id,max( video_digg_count ) as max_date, min( video_digg_count ) as min_date " \
                "from parquet.live_show_video " \
                "WHERE plat =  "+platform_id+" AND create_time >= "+begin_create_time+" AND create_time <  "+end_create_time+" and  date >= '"+begin_date+"'  and  date <  '"+end_date+"' " \
                "GROUP BY room_id,video_id,date) t1 " \
                "GROUP BY room_id,video_id) t2 " \
                "left outer JOIN(SELECT	room_id,tag_id1 AS tag_id,	tag_name " \
                "FROM(SELECT DISTINCT room_id, tag_id AS tag_id1 " \
                "FROM temporary.anchor_tag_ref " \
                "WHERE plat =  "+platform_id+")  a " \
                "LEFT outer JOIN ( " \
                "SELECT  tag_id AS tag_id2, tag_name " \
                "FROM temporary.anchor_tag_config " \
                "WHERE plat =  "+platform_id+" ) b ON a.tag_id1 = b.tag_id2 )t3 on t2.room_id1 = t3.room_id " \
                "WHERE tag_id is not null)t4 GROUP BY tag_id,tag_name"

        return  sql_4

    def date_to_timestamp(self,date):
        #
        timestamp = time.mktime(date.timetuple())
        timestamp = str(timestamp).split(".")[0]
        print(str(date) +"___" + timestamp)
        return timestamp

    def str_to_date_time(self,date_str):
        date_time = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        return date_time

    def add_month_for_date(self,date_str,month_num=0):
        from dateutil.relativedelta import relativedelta
        date_time = self.str_to_date_time(date_str)
        res_date_time = date_time + relativedelta(months=month_num)
        # res_date_time = str(res_date_time)
        return res_date_time

if __name__ == '__main__':
    m=Main()
    months = [8]
    month_num_increase = 2
    type = "video_digg_count"  # video_view_count
    platform_id = "201"

    print("============执行操作=============")
    for month in months:
        print("***************************第 " + str(month) + " 月份******************************")
        next_month = month + 1
        begin_date=""
        if len(str(month)) == 1:
            begin_date = "2019-0" + str(month) + "-01"

        elif len(str(month)) == 2 :
            begin_date = "2019-" + str(month) + "-01"

        end_date = m.add_month_for_date(begin_date,month_num_increase)
        begin_date = m.str_to_date_time(begin_date)

        begin_create_time =str( m.date_to_timestamp(begin_date))
        end_create_time = str(m.date_to_timestamp(end_date))

        print("begin_date=" + str(begin_date))
        print("end_date = "+str(end_date))
        print("begin_create_time=" + str(begin_create_time))
        print("end_create_time = " + end_create_time)
        begin_date = str(begin_date).split(" ")[0].replace("-","")
        end_date = str(end_date).split(" ")[0].replace("-","")
        sql = m.make_sql(platform_id, begin_date,end_date,1001,begin_create_time,end_create_time)
        print(sql)