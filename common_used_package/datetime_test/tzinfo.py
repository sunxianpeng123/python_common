# encoding: utf-8

"""
@author: sunxianpeng
@file: tzinfo.py
@time: 2019/12/1 17:35
"""
from datetime import datetime,timedelta,tzinfo

class UTC(tzinfo):
    def __init__(self,offset=0):
        self._offset = offset
    def utcoffset(self,dt):
        return timedelta(hours=self._offset)
    def tzname(self,dt):
        return "UTC +%s" % self._offset
    def dst(self, dt):
        return timedelta(hours=self._offset)


if __name__ == '__main__':
    # 北京时间
    beijing = datetime(2011, 11, 11, 0, 0, 0, tzinfo=UTC(8))
    print("beijing time:", beijing)
    # 曼谷时间
    bangkok = datetime(2011, 11, 11, 0, 0, 0, tzinfo=UTC(7))
    print("bangkok time", bangkok)
    # 北京时间转成曼谷时间
    print("beijing-time to bangkok-time:", beijing.astimezone(UTC(7)))
    # 计算时间差时也会考虑时区的问题
    timespan = beijing - bangkok
    print("时差:", timespan)
    #beijing time: 2011-11-11 00:00:00+08:00
    # bangkok time 2011-11-11 00:00:00+07:00
    # beijing-time to bangkok-time: 2011-11-10 23:00:00+07:00
    # 时差: -1 day, 23:00:00

