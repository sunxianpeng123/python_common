# encoding: utf-8

"""
@author: sunxianpeng
@file: iris_data.py
@time: 2017/11/10 17:46
"""

from sklearn import datasets

class Main():
    def __init__(self):
        pass


if __name__ == '__main__':
    iris =datasets.load_iris()
    digits=datasets.load_digits
    print type(iris)
    # print iris.values()
    print digits

