# encoding: utf-8

"""
@author: sunxianpeng
@file: datetime_test.py
@time: 2019/12/1 17:12
"""
import time
from datetime import datetime

def static_method_and_attribute():
    print('#####################静态方法和属性######################')
    # datetime.today()：返回一个表示当前本地时间的datetime对象；
    # datetime.now([tz])：返回一个表示当前本地时间的datetime对象，如果提供了参数tz，则获取tz参数所指时区的本地时间；
    # datetime.utcnow()：返回一个当前utc时间的datetime对象；  # 格林威治时间
    # datetime.fromtimestamp(timestamp[, tz])：根据时间戮创建一个datetime对象，参数tz指定时区信息；
    # datetime.utcfromtimestamp(timestamp)：根据时间戮创建一个datetime对象；
    # datetime.combine(date, time)：根据date和time，创建一个datetime对象；
    # datetime.strptime(date_string, format)：将格式字符串转换为datetime对象；
    print('datetime.max = {}'.format(datetime.max))
    print('datetime.min = {}'.format(datetime.min))
    print('datetime.resolution = {}'.format(datetime.resolution))
    print('datetime.today() = {}'.format(datetime.today()))
    print('datetime.now() = {}'.format(datetime.now()))
    print('datetime.utcnow() = {}'.format(datetime.utcnow()))
    print('datetime.fromtimestamp(time.time()) = {}'.format(datetime.fromtimestamp(time.time())))
    print('datetime.utcfromtimestamp(time.time()) = {}'.format(datetime.utcfromtimestamp(time.time())))
    # datetime.max = 9999-12-31 23:59:59.999999
    # datetime.min = 0001-01-01 00:00:00
    # datetime.resolution = 0:00:00.000001
    # datetime.today() = 2019-12-01 17:16:01.729000
    # datetime.now() = 2019-12-01 17:16:01.729000
    # datetime.utcnow() = 2019-12-01 09:16:01.729000
    # datetime.fromtimestamp(time.time()) = 2019-12-01 17:16:01.729000
    # datetime.utcfromtimestamp(time.time()) = 2019-12-01 09:16:01.729000




def dynamic_method_and_attribute():
    print('#####################动态方法和属性######################')
    # dt = datetime.now()  # datetime对象
    # dt.year、month、day、hour、minute、second、microsecond、tzinfo：
    # dt.date()：获取date对象；
    # dt.time()：获取time对象；
    # dt.replace([year[, month[, day[, hour[, minute[, second[, microsecond[, tzinfo]]]]]]]])：
    # dt.timetuple()
    # dt.utctimetuple()
    # dt.toordinal()
    # dt.weekday()
    # dt.isocalendar()
    # dt.isoformat([sep])
    # dt.ctime()：返回一个日期时间的C格式字符串，等效于time.ctime(time.mktime(dt.timetuple()))；
    # dt.strftime(format)
    dt = datetime.now()
    dt2 = dt.replace(year=2020)
    print('dt = {}'.format(dt))
    print('year={},month={},day={},hour={},minute={},second={},microsecond={},tzinfo={}'.format(
        dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second,dt.microsecond,dt.tzinfo
    ))
    print('dt date = {}'.format(dt.date()))
    print('dt.time = {}'.format(dt.time()))
    print('dt2 = {}'.format(dt2))
    print('dt.timetuple = {}'.format(dt.timetuple()))
    print('dt.utctimetuple() = {}')
    print('dt.toordinal() = {}')
    print('dt.weekday() = {}')
    print('dt.isocalendar() = {}')
    print('dt.isoformat() = {}')
    print('dt.ctime() = {}')
    print('dt.strftime() = {}')
    #dt = 2019-12-01 17:26:34.147000
    # year=2019,month=12,day=1,hour=17,minute=26,second=34,microsecond=147000,tzinfo=None
    # dt date = 2019-12-01
    # dt.time = 17:26:34.147000
    # dt2 = 2020-12-01 17:26:34.147000
    # dt.timetuple = time.struct_time(tm_year=2019, tm_mon=12, tm_mday=1, tm_hour=17, tm_min=26, tm_sec=34, tm_wday=6, tm_yday=335, tm_isdst=-1)
    # dt.utctimetuple() = {}
    # dt.toordinal() = {}
    # dt.weekday() = {}
    # dt.isocalendar() = {}
    # dt.isoformat() = {}
    # dt.ctime() = {}
    # dt.strftime() = {}

if __name__ == '__main__':
    static_method_and_attribute()
    dynamic_method_and_attribute()