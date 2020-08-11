# encoding: utf-8

"""
@author: sunxianpeng
@file: 1_a_yuan_linear_demo_v1.py
@time: 2019/11/22 14:22
"""

import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

"""一元线性回归模型"""


def getData():
    x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                        [9.779], [6.182], [7.59], [2.167], [7.042],
                        [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

    y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                        [3.366], [2.596], [2.53], [1.221], [2.827],
                        [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
    # 转换成 tensor
    x_train = torch.from_numpy(x_train)
    # 转换成 tensor
    y_train = torch.from_numpy(y_train)
    return x_train, y_train


def plt_data(x, y, color='bo', label=None):
    plt.plot(x.data.numpy(), y.data.numpy(), color, label=label)
    plt.show()


def plt_data2(x, y, pred):
    plt.plot(x.data.numpy(), pred.data.numpy(), 'bo', label='real')
    plt.plot(x.data.numpy(), y.data.numpy(), 'ro', label='estimated')
    plt.legend()
    plt.show()


def linear_model(x, w, b):
    return x * w + b


def getLoss(pred, y):
    return torch.mean((pred - y) ** 2)


if __name__ == '__main__':
    epochs = 100
    learning_rate = 0.001

    torch.manual_seed(2019)
    x_train, y_train = getData()
    # plt_data(x_train,y_train)

    # 定义参数
    w = torch.randn(1, requires_grad=True)  # 随机初始化
    b = torch.zeros(1, requires_grad=True)  # 使用0进行初始化
    print("w = {}\nw.shape = {}\nb = {}\nb.shape = {}".format(w, w.shape, b, b.shape))

    for epoch in range(epochs):
        pred = linear_model(x_train, w, b)
        loss = getLoss(pred, y_train)

        loss.backward()

        w.data = w.data - learning_rate * w.grad.data
        b.data = b.data - learning_rate * b.grad.data
        if epoch % 1 == 0:
            print('epoch = {},loss = {}'.format(epoch, loss.item()))
        #在使用pytorch实现多项线性回归中，在grad更新时，每一次运算后都需要将上一次的梯度记录清空，
        w.grad.data.zero_()#清空
        b.grad.data.zero_()

    pred = linear_model(x_train, w, b)
    plt_data2(x_train, y_train, pred)
