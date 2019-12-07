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

def plt_decision_boundary(model,x,y):
    #	Set	min	and	max	values	and	give	it	some	padding
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


def get_seq_model():
    seq_net = nn.Sequential(
        nn.Linear(2, 4),
        nn.Tanh(),
        nn.Linear(4, 1)
    )
    # 访问第0层网络
    print(seq_net[0])
    #访问第0层权重
    w0 = seq_net[0].weight
    print("w0 = {}".format(w0))
    return seq_net


def plot_seq(x):
    out	= torch.sigmoid(seq_net(torch.from_numpy(x).float())).data.numpy()
    out	=	(out > 0.5)	* 1
    return out

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

    seq_net = get_seq_model()
    # 获得模型参数
    param =seq_net.parameters()
    optimizer = torch.optim.SGD(param,lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        pred = seq_net(x)
        loss = criterion(pred,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 100 == 0:
            print("epoch :{},loss ={}".format(epoch,loss))
    """画出边界"""
    plt_decision_boundary(lambda x: plot_seq(x), x.numpy(), y.numpy())
    plt.title('sequential')
    plt.show()
    """保存、读取模型"""
    # 两种保存方式
    # 1
    # 将模型架构和参数保存在一起
    save_name = 'save_seq_net.pth'
    torch.save(seq_net,save_name)
    torch.save({'state_dict': seq_net.state_dict()})
    seq_net1 = torch.load(save_name)
    # 2 推荐方式
    # 只保存参数
    param_name = 'save_seq_net_params.pth'
    torch.save({'state_dict': seq_net.state_dict()}, param_name)
    seq_net2 = get_seq_model()
    seq_net2.load_state_dict(torch.load(param_name)['state_dict'])

