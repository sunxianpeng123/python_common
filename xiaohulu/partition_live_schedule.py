# encoding: utf-8

"""
@author: sunxianpeng
@file: partition.py
@time: 2019/2/25 16:52
"""


class Main():
    def __init__(self):
        pass
    def dropPartitions201(self):
        resultArr=[]
        first ="ALTER TABLE parquet.live_show_comment_test ADD IF NOT EXISTS PARTITION(plat=201,date="
        second =",group="
        third=") LOCATION 'hdfs://master20:8020/user/spark/parquet/douyin/comment_v2/plat=201/date="
        fouth = "/group="
        f = open(u"F:\PythonProjects\Test_svm/201_hive_partition_sql.txt","w")

        import datetime
        now = datetime.datetime.now()
        print (now)
        i= -1
        while 1:
            date = now + datetime.timedelta(days= i)
            date = str(date).split(" ")[0].replace("-","")
            if int(date) >= 20160905:
                for group in range(0, 20):
                    s = first + date + second + str(group) + third +date+fouth + str(group) +"';"
                    f.write(s)
                    f.write("\n")
                    resultArr.append(s)

                i -=1
            else:
                break
        f.close()
# ALTER TABLE parquet.label_parquet ADD IF NOT EXISTS PARTITION(plat=201,date=20190527,group=999) LOCATION 'hdfs://master20:8020/tmp/repartition/comment/plat=201/date=20190527/group=999';

    def addPartitionsU(self,table_name,platform_id,table_type):
        resultArr=[]
        one ="ALTER TABLE parquet."+table_name+" ADD IF NOT EXISTS PARTITION(plat="+str(platform_id)+",date="
        two=") LOCATION 'hdfs://master20:8020/user/spark/parquet/anchor_live_schedule/"+table_type+"/plat="+str(platform_id)+"/date="

        f = open(u"F:\PythonProjects\python_study/xiaohulu/data/"+str(platform_id)+"_"+table_type+"_hive_partition_sql.txt","w")
        import datetime
        # now = datetime.datetime.now()#获取当前系统时间
        time_run = "2019-07-08 00:00:00"
        now = datetime.datetime.strptime(time_run, "%Y-%m-%d %H:%M:%S")#指定某个时间
        print (now)
        i= 0
        while 1:
            date = now + datetime.timedelta(days= i)
            date = str(date).split(" ")[0].replace("-","")
            if  int(date) <= 20200731:
                s = one + str(date) + two + str(date) + "';"
                print(s)
                # f.write(s)
                # f.write("\n")
                # resultArr.append(s)
                i +=1
            else:
                break
        f.close()

if __name__ == '__main__':
    m=Main()
    msg_table_name = "live_schedule_message"#
    message_table_type="message"#gift
    gift_table_name = "live_schedule_gift"#
    gift_table_type = "gift"  # gift
    ####message
    all_platform_ids = [1,2,9,15,57,59,36,8,26,12,28,29,60]
    for platform_id in [28,29,60]:
        m.addPartitionsU(msg_table_name, platform_id, message_table_type)  # message
        # m.addPartitionsU(gift_table_name, platform_id, gift_table_type)

