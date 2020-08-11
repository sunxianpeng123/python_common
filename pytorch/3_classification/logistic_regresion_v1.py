# encoding: utf-8

"""
@author: sunxianpeng
@file: logistic_regresion_v1.py
@time: 2019/11/22 20:39
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

import time


def getData():
    data = []
    with open('logistic.txt') as f:
        data_list = [i.split('\n')[0].split(',') for i in f.readlines()]
        data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]
    # 标准化
    x0_max = max([i[0] for i in data])
    x1_max = max([i[1] for i in data])
    data = [(i[0] / x0_max, i[1] / x1_max, i[2]) for i in data]
    return data


def plt_data(data):
    # 取出不同分类标签的点
    x0 = list(filter(lambda x: x[-1] == 0.0, data))  # 选择第一类的点
    x1 = list(filter(lambda x: x[-1] == 1.0, data))  # 选择低二类的点
    plt_x0 = [i[0] for i in x0]
    plt_y0 = [i[1] for i in x0]
    plt_x1 = [i[0] for i in x1]
    plt_y1 = [i[1] for i in x1]

    plt.plot(plt_x0, plt_y0, 'ro', label='x_0')
    plt.plot(plt_x1, plt_y1, 'bo', label='x_1')
    plt.legend(loc='best')
    plt.show()


def plt_sigmoid():
    x = np.arange(-10, 10.01, 0.01)
    y = sigmoid(x)
    plt.plot(x, y, 'r')
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logistic_regression(x, w, b):
    return F.sigmoid(torch.mm(x, w) + b)


def getLoss(pred, y):
    # torch.clamp(input, min, max, out=None) → Tensor
    # 将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。
    #       | min, if x_i < min
    # y_i = | x_i, if min <= x_i <= max
    #       | max, if x_i > max
    # 如果输入是FloatTensor or DoubleTensor类型，则参数min max 必须为实数，否则须为整数。【译注：似乎并非如此，无关输入类型，min， max取整数、实数皆可。】
    # 参数：
    #     input (Tensor) – 输入张量
    #     min (Number) – 限制范围下限
    #     max (Number) – 限制范围上限
    #     out (Tensor, optional) – 输出张量
    logits = (y * pred.clamp(1e-12).log() + (1 - y) * (1 - pred).clamp(1e-12).log()).mean()
    return -logits


def plt_after_training_model(data, w, b):
    w0 = w[0].item()
    w1 = w[1].item()
    b0 = b.item()
    # 取出不同分类标签的点
    x0 = list(filter(lambda x: x[-1] == 0.0, data))  # 选择第一类的点
    x1 = list(filter(lambda x: x[-1] == 1.0, data))  # 选择低二类的点
    plot_x0 = [i[0] for i in x0]
    plot_y0 = [i[1] for i in x0]
    plot_x1 = [i[0] for i in x1]
    plot_y1 = [i[1] for i in x1]

    plot_x = np.arange(0.2, 1, 0.01)
    plot_y = (-w0 * plot_x - b0) / w1

    plt.plot(plot_x, plot_y, 'g', label='cutting	line')
    plt.plot(plot_x0, plot_y0, 'ro', label='x_0')
    plt.plot(plot_x1, plot_y1, 'bo', label='x_1')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(2019)
    epochs = 1000
    learning_rate = 1
    data1 = getData()
    # plt_data(data1)
    data = np.array(data1, dtype='float32')
    x_data = torch.from_numpy(data[:, 0:2])
    y_data = torch.from_numpy(data[:, -1]).unsqueeze(1)  # 打平,转换成[100,1]
    # torch.from_numpy(data[:, -1]).shape = torch.Size([100])
    print("torch.from_numpy(data[:, -1]).shape = {}".format(torch.from_numpy(data[:, -1]).shape))
    # torch.from_numpy(data[:, -1]).unsqueeze(1) shape = torch.Size([100, 1])
    print("torch.from_numpy(data[:, -1]).unsqueeze(1) shape = {}".format(
        torch.from_numpy(data[:, -1]).unsqueeze(1).shape))

    w = torch.randn(2, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    w = torch.nn.Parameter(w)
    b = torch.nn.Parameter(b)

    optimizer = torch.optim.SGD([w, b], lr=learning_rate)

    start_time = time.time()

    for epoch in range(epochs):
        pred = logistic_regression(x_data, w, b)
        loss = getLoss(pred, y_data)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # https://blog.csdn.net/ljs_a/article/details/79848994
        # ge gt le 张量的比较函数,
        mask = pred.ge(0.5).float()
        print(mask)
        acc = (mask == y_data).sum().item() / y_data.shape[0]
        if epoch % 200 == 0:
            print('epoch:{},Loss:{:.5f},Acc:{:.5f}'.format(epoch + 1, loss.item(), acc))
            print("     mask shape = {}".format(mask.shape))
    during = time.time() - start_time
    print('During	Time:	{:.3f}	s'.format(during))

    plt_after_training_model(data, w, b)
