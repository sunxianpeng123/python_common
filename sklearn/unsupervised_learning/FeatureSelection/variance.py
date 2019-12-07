# encoding: utf-8

"""
@author: sunxianpeng
@file: variance.py
@time: 2017/11/16 14:41
"""

from sklearn.feature_selection import  VarianceThreshold
class Main():
    #数据的差别大小
    def __init__(self):
        pass
    def variance(self):
        X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
        sel =VarianceThreshold(threshold=0.8*(1-0.8))
        # sel.fit(X)#fit方法是用于从一个训练集中学习模型参数，其中就包括了归一化时用到的均值，标准偏差。
        print sel.fit_transform(X)#即fit_transform()的作用就是先拟合数据，然后转化它将其转化为其他需要形式

if __name__ == '__main__':
    m=Main()
    m.variance()