# encoding: utf-8

"""
@author: sunxianpeng
@file: 1_helloworld_1.py
@time: 2019/11/6 17:45
"""


import numpy as np
import torch


if __name__ == '__main__':
    """说只要定义了forward pass(前向神经网络)，计算了loss之后，
    PyTorch可以自动求导计算模型所有参数的梯度。"""
    N,D_in,H,D_out = 64,1000,100,10
    # 随机创建一些训练数据
    x = torch.randn(N,D_in)
    y = torch.randn(N,D_out)
    w1 = torch.randn(D_in,H,requires_grad=True)
    w2 = torch.randn(H,D_out,requires_grad=True)

    learning_rate = 10**-6
    for it in range(500):
        # forward pass
        y_pred = x.mm(w1).clamp(min=0).mm(w2)
        # compute loss
        loss = (y_pred - y).pow(2).sum()
        print(" 第 {} 次 前向传播结束，总体损失为 {:.10f}".format(it,loss.item()))
        # Backward pass
        loss.backward()
        # 更新参数
        with torch.no_grad():
            # 注意：使用w1 = w1 -  learning_rate * w1.grad形式时，w1.grad.zero_()会报错，和内存分配有关
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad
            # 重置梯度
            w1.grad.zero_()
            w2.grad.zero_()





