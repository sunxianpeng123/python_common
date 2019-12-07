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
    return x_train,y_train

def plt_data(x, y, color='bo', label=None):
    plt.plot(x.data.numpy(),y.data.numpy(),color,label=label)
    plt.show()

def plt_data2(x,y,pred):
    plt.plot(x.data.numpy(), pred.data.numpy(), 'bo', label='real')
    plt.plot(x.data.numpy(), y.data.numpy(), 'ro', label='estimated')
    plt.legend()
    plt.show()

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1,1)
    def forward(self,x):
        pred = self.linear(x)
        return pred

if __name__ == '__main__':
    epochs = 1000
    learning_rate = 0.001

    torch.manual_seed(2019)
    x_train,y_train = getData()

    model = Model()
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

    for epoch in range(epochs):
        pred = model(x_train)
        loss = criterion(pred,y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 100 == 0:
            print('epoch = {},loss = {}'.format(epoch,loss.item()))

    pred = model(x_train)
    plt_data2(x_train, y_train, pred)






