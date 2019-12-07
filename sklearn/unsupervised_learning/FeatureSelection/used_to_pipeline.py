# encoding: utf-8

"""
@author: sunxianpeng
@file: used_to_pipeline.py
@time: 2017/11/16 15:43
"""
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
class Main():
    def __init__(self):
        pass


if __name__ == '__main__':
    iris = load_iris()
    X, y = iris.data, iris.target
    clf = Pipeline([
        ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
        ('classification', RandomForestClassifier())
    ])
    clf.fit(X, y)