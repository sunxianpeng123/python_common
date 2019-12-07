# encoding: utf-8

"""
@author: sunxianpeng
@file: timedelta.py
@time: 2019/12/1 17:27
"""
from datetime import datetime
from datetime import timedelta

def datetime_add_or_minus():
    dt = datetime.now()
    # 天数减一天
    dt_minus_1 = dt + timedelta(days=-1)
    dt_minus_2 = dt - timedelta(days=1)
    #计算两个日期的差值
    diff = dt - dt_minus_1
    print('dt = {}'.format(dt))
    print('dt_minus_1 = {}'.format(dt_minus_1))
    print('dt_minus_2 = {}'.format(dt_minus_2))
    print('diff = {}'.format(diff))
    print('diff days = {}'.format(diff.days))
    print('diff total_seconds = {}'.format(diff.total_seconds()))
    #dt = 2019-12-01 17:33:58.560000
    # dt_minus_1 = 2019-11-30 17:33:58.560000
    # dt_minus_2 = 2019-11-30 17:33:58.560000
    # diff = 1 day, 0:00:00
    # diff days = 1
    # diff total_seconds = 86400.0

if __name__ == '__main__':
    datetime_add_or_minus()

