# encoding: utf-8

"""
@author: sunxianpeng
@file: test.py
@time: 2020/4/20 16:39
"""
from math import log # 自然对数为底

class Main():
    def __init__(self):
        pass


if __name__ == '__main__':
    y_true =[0]
    y_pred = [[0.9, 0.1]]
    Loss = 0
    for label, prob in zip(y_true, y_pred):
        print("===label = {},, prob = {}=======".format(label,prob))
        Loss -= ( (1 - label) * log(prob[1]))
        print(Loss)
    Loss = Loss / len(y_true)
    print('Loss by equation: %s.' % Loss)