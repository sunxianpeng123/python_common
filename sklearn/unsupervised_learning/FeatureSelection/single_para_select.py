# encoding: utf-8

"""
@author: sunxianpeng
@file: single_para_select.py
@time: 2017/11/16 14:52
"""
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

class Main():
    #https://segmentfault.com/a/1190000003719712            卡方
    def __init__(self):
        pass


if __name__ == '__main__':
    iris = load_iris()
    X, y = iris.data, iris.target
    print X.shape
    X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
    print X_new.shape