# encoding: utf-8

"""
@author: sunxianpeng
@file: SelectFromModel.py
@time: 2017/11/16 15:25
"""
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

class Main():
    def __init__(self):
        pass

    def select_feature_L1(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        print X.shape
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
        model = SelectFromModel(lsvc, prefit=True)
        X_new = model.transform(X)
        print X_new.shape

    def select_feature_tree(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        print X.shape
        clf = ExtraTreesClassifier()
        clf = clf.fit(X, y)
        print clf.feature_importances_
        model = SelectFromModel(clf, prefit=True)
        X_new = model.transform(X)
        print X_new.shape



if __name__ == '__main__':
    m=Main()
    m.select_feature_L1()
    m.select_feature_tree()