# encoding: utf-8

"""
@author: sunxianpeng
@file: Classification.py
@time: 2017/11/16 11:07
"""

from sklearn.linear_model import SGDClassifier
class Main():
    def __init__(self):
        pass


if __name__ == '__main__':
    X = [[0., 0.], [1., 1.]]
    y = [0, 1]
    clf=SGDClassifier(loss="hinge", penalty="l2")#loss functions ( 损失函数 ) 和 penalties for classification ( 分类处罚 )。
    clf.fit(X, y)
    print clf.predict([[2., 2.]])
    print clf.coef_ #SGD 拟合训练数据的线性模型。成员 coef_ 保存模型参数：
    print clf.intercept_#成员 intercept_ 保持 intercept ( 截距 )（aka offset 或 bias ）：
    print clf.decision_function([[2., 2.]])#模型是否应该使用 intercept ( 截距 ) ，即 biased hyperplane ( 偏置的超平面 ) ，由参数 fit_intercept 控制。