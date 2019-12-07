# encoding: utf-8

"""
@author: sunxianpeng
@file: classify.py
@time: 2017/11/16 15:56
"""

from sklearn.neural_network import MLPClassifier
class Main():
    #多层感知器（MLP）算法
    def __init__(self):
        pass


if __name__ == '__main__':
    X = [[0., 0.], [1., 1.]]
    y = [0, 1]
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (5, 2), random_state = 1)
    clf.fit(X, y)
    print clf.predict([[1., 2.]])
    print clf.predict([[0., 0.]])