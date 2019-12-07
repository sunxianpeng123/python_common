# encoding: utf-8

"""
@author: sunxianpeng
@file: time_test.py
@time: 2019/12/1 17:02
"""
from datetime import time
def static_method_and_attribute():
    print('#####################静态方法和属性######################')
    # time.min、time.max：time类所能表示的最小、最大时间。其中，time.min = time(0, 0, 0, 0)， time.max = time(23, 59, 59, 999999)；
    # time.resolution：时间的最小单位，这里是1微秒；
    print('time max = {}'.format(time.max))
    print('time min = {}'.format(time.min))
    print('time resolution = {}'.format(time.resolution))
    #time max = 23:59:59.999999
    # time min = 00:00:00
    # time resolution = 0:00:00.000001


def dynamic_method_and_attribute():
    print('#####################动态方法和属性######################')
    # t1 = datetime.time(10, 23, 15)  # time对象
    # t1.hour、t1.minute、t1.second、t1.microsecond：时、分、秒、微秒；
    # t1.tzinfo：时区信息；
    # t1.replace([hour[, minute[, second[, microsecond[, tzinfo]]]]] )：创建一个新的时间对象，用参数指定的时、分、秒、微秒代替原有对象中的属性（原有对象仍保持不变）；
    # t1.isoformat()：返回型如
    # "HH:MM:SS"
    # 格式的字符串表示；
    # t1.strftime(fmt)：同time模块中的format；
    t1 = time(16,23,15)
    t2 = t1.replace(hour=23)
    print('t1 = {}'.format(t1))
    print('hout= {},minute ={},second ={},microsecond={}'.format(t1.hour,t1.minute,t1.second,t1.microsecond))
    print('t2 = {}'.format(t2))
    print('isoformat = {}'.format(t1.isoformat()))
    print('strftime = {}'.format(t1.strftime('%X')))
    # t1 = 16:23: 15
    # hout = 16, minute = 23, second = 15, microsecond = 0
    # t2 = 23:23: 15
    # isoformat = 16:23: 15
    # strftime = 16:23: 15

if __name__ == '__main__':
    static_method_and_attribute()
    dynamic_method_and_attribute()