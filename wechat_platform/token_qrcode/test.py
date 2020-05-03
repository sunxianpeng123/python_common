# encoding: utf-8

"""
@author: sunxianpeng
@file: test.py
@time: 2020/5/2 21:38
"""

from Common import *

class Main():
    def __init__(self):
        pass


if __name__ == '__main__':
    c = Common()
    url = "http://testwechat.applinzi.com/"
    res = c.requestGet(url).text
    print(res)