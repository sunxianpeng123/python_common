
# encoding: utf-8

"""
@author: sunxianpeng
@file: demo.py
@time: 2019/12/3 16:05
"""


class Main():
    def __init__(self):
        pass


if __name__ == '__main__':
    sql = 'select * from table_name where column in '
    list = ['王者荣耀','刺激战场']
    sql = sql + "('" + "','".join(list) + "')"
    print(sql)