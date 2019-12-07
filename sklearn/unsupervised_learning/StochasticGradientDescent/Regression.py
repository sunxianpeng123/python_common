# encoding: utf-8

"""
@author: sunxianpeng
@file: Regression.py
@time: 2017/11/16 11:15
"""
from sklearn.preprocessing import StandardScaler

class Main():
    def __init__(self):
        pass


if __name__ == '__main__':
    # 随机梯度下降对特征缩放很敏感，因此强烈建议您扩展数据。例如，将输入向量 X 上的每个属性缩放到 [0,1] 或 [-1，+ 1] ，
    # 或将其标准化为平均值 0 和方差 1 . 注意，必须将相同的缩放应用于测试向量获得有意义的结果。这可以使用 StandardScaler 轻松完成：
    scaler = StandardScaler()
    # scaler.fit(X_train)  # Don't cheat - fit only on training data
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)  # apply same transformation to test data