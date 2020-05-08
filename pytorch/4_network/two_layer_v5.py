# encoding: utf-8

"""
@author: sunxianpeng
@file: 1_helloworld_1.py
@time: 2019/11/6 17:45
"""


import numpy as np
import torch
import torch.nn as nn
class TwoLayerNet(torch.nn.Module):
    def __init__(self,D_in,H,D_out):
        super(TwoLayerNet,self).__init__()
        # 定义模型结构
        self.linear1 = torch.nn.Linear(D_in,H,bias=False)
        self.linear2 = torch.nn.Linear(H,D_out,bias=False)

    def forward(self, x):
        layer1 = self.linear1(x)
        layer1_relu = layer1.clamp(min=0)
        y_pred = self.linear2(layer1_relu)
        return y_pred

if __name__ == '__main__':
    """自定义 nn Modules
    模型继承自nn.Module类。如果需要定义一个比Sequential模型更加复杂的模型，就需要定义nn.Module模型。"""
    N,D_in,H,D_out = 64,1000,100,10
    # 随机创建一些训练数据
    x = torch.randn(N,D_in)
    y = torch.randn(N,D_out)

    # 模型
    model = TwoLayerNet(D_in,H,D_out)
    # 方差损失
    loss_fn = nn.MSELoss(reduction="sum")
    #参数更新，优化算法
    learning_rate = 10**-4
    print("learning_rate = {}".format(learning_rate))
    # optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for it in range(500):
        # forward pass
        y_pred = model(x)
        # compute loss
        loss = loss_fn(y_pred,y)
        print(" 第 {} 次 前向传播结束，总体损失为 {:.10f}".format(it,loss.item()))
        #
        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        # update weights of w1 and w2
        optimizer.step()






