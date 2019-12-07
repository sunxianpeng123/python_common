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

def getData():
    # test_func()#测试多项式模型
    w_target = np.array([0.5, 3, 2.4])  #
    b_target = np.array([0.9])  #
    x_sample = np.arange(-3, 3.1, 0.1)
    y_sample = b_target[0] + w_target[0] * x_sample + w_target[1] * x_sample ** 2 + w_target[2] + x_sample ** 3

    x_train = np.stack([x_sample ** i for i in range(1,4)],axis=1)# n * 3
    y_train = torch.from_numpy(y_sample).double().unsqueeze(1)
    x_train = torch.from_numpy(x_train).float()
    return x_train,y_train,y_sample

def test_func():
    w_target = np.array([0.5, 3, 2.4])  #
    b_target = np.array([0.9])  # ਧԎ݇හ
    f_des = 'y	=	{:.2f}	+	{:.2f}	*	x	+	{:.2f}	*	x^2	+	{:.2f}	*	x^3'.format(
        b_target[0], w_target[0], w_target[1], w_target[2])  # 打印函数公式
    print(f_des)

    x_sample = np.arange(-3, 3.1, 0.1)
    y_sample = b_target[0] + w_target[0] * x_sample + w_target[1] * x_sample ** 2 + w_target[2] + x_sample ** 3
    plt_data(x_sample, y_sample)

def plt_data(x_train, y,pred):
    plt.plot(x_train.data.numpy()[:,0],pred.data.numpy(),label='fitting curve',color='r')
    plt.plot(x_train.data.numpy()[:,0],y,label='real curve',color='b')
    plt.legend()
    plt.show()

def multi_linear(x,w,b):
    return torch.mm(x,w) + b

def getLoss(pred,y):
    return torch.mean((pred - y) ** 2)

if __name__ == '__main__':
    torch.manual_seed(2019)
    epochs = 100
    learning_rate = 0.001

    x_train,y_train,y_sample = getData()

    w = torch.randn(3,1,requires_grad=True).float()
    b = torch.zeros(1,requires_grad=True).float()
    print("w = {}\nw.shape = {}\nb = {}\nb.shape = {}".format(w,w.shape,b,b.shape))

    for epoch in  range(epochs):
        pred = multi_linear(x_train,w,b)
        loss = getLoss(pred,y_train)


        loss.backward()

        w.data = w.data - learning_rate * w.grad.data
        b.data = b.data - learning_rate * b.grad.data
        w.grad.data.zero_()
        b.grad.data.zero_()
        if epoch % 10 ==0:
            print("w = {}\nw.shape = {}\nb = {}\nb.shape = {}".format(w, w.shape, b, b.shape))

    pred = multi_linear(x_train,w,b)
    plt_data(x_train,y_sample,pred)









