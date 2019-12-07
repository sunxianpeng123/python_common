# encoding: utf-8

"""
@author: sunxianpeng
@file: GaussianNaiveBayes.py
@time: 2017/11/16 12:42
"""

from sklearn import  datasets
from sklearn.naive_bayes import GaussianNB
class Main():
    def __init__(self):
        pass


if __name__ == '__main__':
    iris =datasets.load_iris()
    gnb=GaussianNB()
    y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
    print("Number of mislabeled points out of a total %d points : %d"
          % (iris.data.shape[0], (iris.target != y_pred).sum()))
