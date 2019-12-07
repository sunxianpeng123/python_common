# #coding:utf-8
#
# import requests
# import MySQLdb
#
# def download(url,s):
#     print(url)
#     res = requests.get(url)
#     res.raise_for_status()
#     playFile = open(s+'.ts', 'wb')
#     for chunk in res.iter_content(100000):
#         playFile.write(chunk)
#     playFile.close()
#
# if __name__=='__main__':
#     conn= MySQLdb.connect(
#         host='113.107.166.14',
#         port =10006,
#         user='xiangjia_local',
#         passwd='3nhumzW9AZivzWQx',
#         db ='live_platform',
#     )
#
#     s='F:/video'
#
#     cur=conn.cursor()
#     sql="SELECT  id, video_url  from  live_platform.video_statistics  where  statistics_date>='2017-04-17' and video_url is not null and CHAR_LENGTH(messageJson)>100 limit 10"
#     data=cur.execute(sql)
#     info=cur.fetchall()
#     for tup in info:
#         id=tup[0]
#         video_url='http://dv.xiaohulu.com/resource'+tup[1]
#         download(video_url,s)
