# encoding: utf-8

"""
@author: sunxianpeng
@file: regression.py
@time: 2017/11/16 13:13
"""

from sklearn import tree

class Main():
    def __init__(self):
        pass
    def indoor(self):
        X = [[0, 0], [2, 2]]
        y = [0.5, 2.5]
        clf = tree.DecisionTreeRegressor()
        clf = clf.fit(X, y)
        print clf.predict([[1, 1]])

if __name__ == '__main__':
    m=Main()
    m.indoor()