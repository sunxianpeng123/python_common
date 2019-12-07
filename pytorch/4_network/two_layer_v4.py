# encoding: utf-8

"""
@author: sunxianpeng
@file: demo.py
@time: 2019/11/6 17:45
"""


import numpy as np
import torch
import torch.nn as nn

if __name__ == '__main__':
    """不再手动更新模型的weights,而是使用optim这个包来帮助我们更新参数。
     optim这个package提供了各种不同的模型优化方法，包括SGD+momentum, RMSProp, Adam等等。"""
    N,D_in,H,D_out = 64,1000,100,10
    # 随机创建一些训练数据
    x = torch.randn(N,D_in)
    y = torch.randn(N,D_out)
    """ 利用torch nn 建模过程"""
    # 模型
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in,H,bias=False),# w_1 * x + b_1,第一层
        torch.nn.ReLU(),
        torch.nn.Linear(H,D_out,bias=False)# 第二层
    )
    # 权重初始化
    torch.nn.init.normal_(model[0].weight)
    torch.nn.init.normal_(model[2].weight)
    # 设置运行在cpu还是gpu
    # model = model.cuda()# 可以将整个模型转换成gpu上运行的格式
    # 方差损失
    loss_fn = nn.MSELoss(reduction="sum")
    #参数更新，优化算法
    learning_rate = 10**-6
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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






