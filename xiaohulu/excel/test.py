# encoding: utf-8

"""
@author: sunxianpeng
@file: test.py
@time: 2021/1/25 19:05
"""
import os

import pandas as pd
import requests


def function(a, b):
    if a == b:
        return True
    else:
        return False


if __name__ == '__main__':
    path = "2.xlsx"
    df1 = pd.read_excel(path)

    df2 = df1
    allDF = pd.merge(df1, df2, on=['日期', '话题名称', '类型'], how="outer")

    print(allDF.index)  # Int64Index([0, 1, 2, 3], dtype='int64')
    print(allDF.columns)  # Index(['A', 'B', 'C', 'D', 'E', 'F'], dtype='object')

    allDF['新创视频数'] = allDF['新创视频数_x'] - allDF['新创视频数_y']
    allDF['新创视频点赞合计'] = allDF['新创视频点赞合计_x'] - allDF['新创视频点赞合计_y']
    allDF['新创视频评论合计'] = allDF['新创视频评论合计_x'] - allDF['新创视频评论合计_y']
    print(allDF.columns)
    print(allDF)
