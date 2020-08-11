# encoding: utf-8

"""
@author: sunxianpeng
@file: multi_network_v1.py
@time: 2019/11/23 13:20
"""

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def plt_decision_boundary(model, x, y):
    #Set min and	max	values	and	give	it	some	padding
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    h = 0.01
    #	Generate	a	grid	of	points	with	distance	h	between	them
    # 生成网格点的坐标矩阵，
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #	Predict	the	function value	for	the	whole	grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    #	Plothe	contour	and	training	examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(x[:, 0], x[:, 1], c=y.reshape(-1), s=40, cmap=plt.cm.Spectral)


def two_network(x, w1, w2, b1, b2):
    r = torch.mm(x, w1) + b1
    r = F.tanh(r)
    r = torch.mm(r, w2) + b2
    return r


def plot_network(x):
    x = torch.from_numpy(x).float()
    x1 = torch.mm(x, w1) + b1
    x1 = F.tanh(x1)
    x2 = torch.mm(x1, w2) + b2
    out = F.sigmoid(x2)
    out = (out > 0.5) * 1
    print("out = {}".format(out))
    return out.data.numpy()


if __name__ == '__main__':

    np.random.seed(1)
    m = 400  # 样本数量
    N = int(m / 2)  # 每类点的个数
    D = 2  # 维度
    x = np.zeros((m, D))
    y = np.zeros((m, 1), dtype='uint8')  # label 向量，0表示红色，1表示蓝色
    a = 4
    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        x[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    # 画出图像
    plt.scatter(x[:, 0], x[:, 1], c=y.reshape(-1), s=40, cmap=plt.cm.Spectral)
    plt.show()

    # 定义两层神经网络
    epochs = 1000
    learning_rate = 1

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    w1 = nn.Parameter(torch.randn(2, 4, requires_grad=True) * 0.01).float()
    b1 = nn.Parameter(torch.zeros(4)).float()
    w2 = nn.Parameter(torch.randn(4, 1) * 0.01).float()
    b2 = nn.Parameter(torch.zeros(1)).float()
    optimizer = torch.optim.SGD([w1, w2, b1, b2], lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        pred = two_network(x, w1, w2, b1, b2)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 100 == 0:
            print("epoch :{},loss ={}".format(epoch, loss))

    plt_decision_boundary(lambda x: plot_network(x), x.numpy(), y.numpy())
    plt.title('2 layer network')
    plt.show()
