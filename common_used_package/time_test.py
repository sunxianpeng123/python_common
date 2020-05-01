# encoding: utf-8

"""
@author: sunxianpeng
@file: time_test.py
@time: 2019/12/1 16:02
"""
import time

def between_struct_time_and_timestamp():
    print("#####################struct_time 和 timestamp 之间的转换#####################")
    print('************生成timestamp**************')
    # 1 生成timestamp
    print('1 timestamp = {}'.format(time.time()))#1 timestamp = 1575188363.801
    print('************struct_time to timestamp**************')
    # 2 struct_time to timestamp
    # struct_time = time.struct_time(tm_year=2019, tm_mon=12, tm_mday=1, tm_hour=16, tm_min=17, tm_sec=10, tm_wday=6,
    #                                tm_yday=335, tm_isdst=0)
    localtime = time.localtime()
    timestamp = time.mktime(localtime)
    print('2 struct_time = {}'.format(localtime))
    print('2 timestamp = {}'.format(timestamp))#timestamp = 1575188363.0
    print('************timestamp to struct_time 本地时间**************')
    #3 生成 struct_time，timestamp to struct_time 本地时间
    localtime = time.localtime(time.time())
    # struct_time = time.struct_time(tm_year=2019, tm_mon=12, tm_mday=1, tm_hour=16, tm_min=17, tm_sec=10, tm_wday=6,
                                   # tm_yday=335, tm_isdst=0)
    print('3 struct_time = {}'.format(localtime))
    print('************timestamp to struct_time 格林威治时间**************')
    gmtime = time.gmtime()
    # time.struct_time(tm_year=2019, tm_mon=12, tm_mday=1, tm_hour=8, tm_min=23, tm_sec=52, tm_wday=6, tm_yday=335,
    #                  tm_isdst=0)
    print('gmtime = {}'.format(gmtime))
    print('格林威治时间 = {}'.format(time.gmtime(time.time())))

def between_format_time_and_struct_time():
    print("#####################format_time 和 struce_time 之间的转换#####################")
    print('************format_time to struct_time**************')
    format_time = '2019-12-01 16:37:06'
    format = '%Y-%m-%d %X'

    struct_time = time.strptime(format_time,format)
    print('1 format time = {}'.format(format_time))
    print('1 struct time = {}'.format(struct_time))
    # 1 format time = 2019-12-01 16:37:06
    # 1 structtime = time.struct_time(tm_year=2019, tm_mon=12, tm_mday=1, tm_hour=16, tm_min=37, tm_sec=6, tm_wday=6, tm_yday=335,
    #                         tm_isdst=-1)
    print('************struct_time to format_time**************')
    format_time_1 = time.strftime(format)
    format_time_2 = time.strftime(format,time.localtime())
    print('2 format_time_1 = {}'.format(format_time_1))
    print('2 format_time_2 = {}'.format(format_time_2))
    # 2 format_time_1 = 2019-12-01 16:42:27
    # 2 format_time_2 = 2019-12-01 16:42:27

def other_time_op():
    print("#####################其他操作#####################")
    print('************生成固定格式的时间表示格式**************')
    # 生成固定格式的时间表示格式
    t1 = time.asctime(time.localtime())#Sun Dec  1 16:39:50 2019
    t2 = time.ctime(time.time())#Sun Dec  1 16:39:50 2019
    print('1 t1 = {}'.format(t1))
    print('1 t2 = {}'.format(t2))
    # 1 t1 = Sun Dec  1 16:44:57 2019
    # 1 t2 = Sun Dec  1 16:44:57 2019
    print(time.strptime(t1))
    print('************time加减**************')
    # timestamp加减单位以秒为单位
    t3 = time.time()
    t4 = t3 + 10
    print('2 t3 = {}'.format(t3))
    print('2 t4 = {}'.format(t4))
    #2 t3 = 1575189897.4310002
    # 2 t4 = 1575189907.4310002

def add():
    "字符串格式转struct_time格式时间"
    print(time.strptime('Sat Feb 04 14:06:42 2017'))
if __name__ == '__main__':
    between_struct_time_and_timestamp()
    between_format_time_and_struct_time()
    other_time_op()
    add()
