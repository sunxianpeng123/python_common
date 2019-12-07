# encoding: utf-8

"""
@author: sunxianpeng
@file: date_test.py
@time: 2019/12/1 16:50
"""
import time
from datetime import date

def static_method_and_attribute():
    print('#####################静态方法和属性######################')
    # date.max、date.min：date对象所能表示的最大、最小日期；
    # date.resolution：date对象表示日期的最小单位。这里是天。
    # date.today()：返回一个表示当前本地日期的date对象；
    # date.fromtimestamp(timestamp)：根据给定的时间戮，返回一个date对象；
    print('date.max = {}'.format(date.max))
    print('date.min = {}'.format(date.min))
    print('date.today = {}'.format(date.today()))
    print('date.fromtimestamp = {}'.format( date.fromtimestamp(time.time())))
    # date.max = 9999 - 12 - 31
    # date.min = 0001 - 01 - 01
    # date.today = 2019 - 12 - 01
    # date.fromtimestamp = 2019 - 12 - 01

def dynamic_method_and_attribute():
    print('#####################动态方法和属性######################')
    # d1 = date(2011, 06, 03)  # date对象
    # d1.year、date.month、date.day：年、月、日；
    # d1.replace(year, month, day)：生成一个新的日期对象，用参数指定的年，月，日代替原有对象中的属性。（原有对象仍保持不变）
    # d1.timetuple()：返回日期对应的time.struct_time对象；
    # d1.weekday()：返回weekday，如果是星期一，返回0；如果是星期2，返回1，以此类推；
    # d1.isoweekday()：返回weekday，如果是星期一，返回1；如果是星期2，返回2，以此类推；
    # d1.isocalendar()：返回格式如(year，month，day)的元组；
    # d1.isoformat()：返回格式如'YYYY-MM-DD’的字符串；
    # d1.strftime(fmt)：和time模块format相同。
    now = date(2019,12,1)
    tomorrow = now.replace(day=2)
    print('now date = {}'.format(now))#now date = 2019-12-01
    print('tomorrow date = {}'.format(tomorrow))  # tomorrow date = 2019-12-02
    print('timetuple = {}'.format(now.timetuple()))
    print('weekday = {}'.format(now.weekday()))
    print('isoweekday = {}'.format(now.isoweekday()))
    print('isocalendar = {}'.format(now.isocalendar()))
    print('isoformat = {}'.format(now.isoformat()))
    print('strftime = {}'.format(now.strftime('%Y-%m-%d')))
    #now date = 2019-12-01
    # tomorrow date = 2019-12-02
    # timetuple = time.struct_time(tm_year=2019, tm_mon=12, tm_mday=1, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=6, tm_yday=335, tm_isdst=-1)
    # weekday = 6
    # isoweekday = 7
    # isocalendar = (2019, 48, 7)
    # isoformat = 2019-12-01
    # strftime = 2019-12-01

if __name__ == '__main__':
    static_method_and_attribute()
    dynamic_method_and_attribute()