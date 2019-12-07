# encoding: utf-8

"""
@author: sunxianpeng
@file: random_forest.py
@time: 2017/11/16 13:40
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
class Main():
    def __init__(self):
        pass
    def randomforest_two(self):
        X = [[0, 0], [1, 1]]
        Y = [0, 1]
        clf = RandomForestClassifier(n_estimators=10)
        clf = clf.fit(X, Y)

    def randomforest_extreme(self):
        X, y = make_blobs(n_samples=10000, n_features=10, centers=100,random_state = 0)
        clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state = 0)
        scores = cross_val_score(clf, X, y)
        print scores.mean()
        clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split = 2, random_state = 0)
        scores = cross_val_score(clf, X, y)
        print(scores.mean())
        clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split = 2, random_state = 0)
        scores = cross_val_score(clf, X, y)
        print scores.mean() > 0.999
        #使用这些方法时要调整的主要参数是n_estimators 和max_features。前者是森林里的树木数量。越大越好，而且计算时间
        # 越长。此外，请注意，超过关键数量的树木，结果将停止显着改善。后者是分割节点时要考虑的特征的随机子集的大小。

if __name__ == '__main__':
    m=Main()
    m.randomforest_two()
    m.randomforest_extreme()
