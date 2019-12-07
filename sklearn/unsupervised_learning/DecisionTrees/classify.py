# encoding: utf-8

"""
@author: sunxianpeng
@file: classify.py
@time: 2017/11/16 12:46
"""
from sklearn import tree
from sklearn.datasets import load_iris
import os
import pydotplus
class Main():
    def __init__(self):
        pass

    def decisionTreeClassifier(self):
        X = [[0, 0], [1, 1]]
        Y = [0, 1]
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X, Y)
        print clf.predict([[2., 2.]])

    def decisionTreeClassifier_iris(self):
        iris = load_iris()
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(iris.data, iris.target)
        with open("iris.dot", 'w') as f:
            f = tree.export_graphviz(clf, out_file=f)
            # os.unlink('iris.dot')
        # with open("iris.pdf", 'w') as f:
        #     dot_data = tree.export_graphviz(clf, out_file=None)
        #     graph = pydotplus.graph_from_dot_data(dot_data)
        #     graph.write_pdf("iris.pdf")
        print clf.predict(iris.data[:1, :])#安装后，可以使用该模型来预测样本类别：
        print clf.predict_proba(iris.data[:1, :])#或者，可以预测每个类的概率，这是叶中相同类的训练样本的分数：

if __name__ == '__main__':
    m=Main()
    m.decisionTreeClassifier()
    m.decisionTreeClassifier_iris()