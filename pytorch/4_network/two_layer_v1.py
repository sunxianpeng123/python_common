# encoding: utf-8

"""
@author: sunxianpeng
@file: demo.py
@time: 2019/11/6 17:45
"""


import numpy as np
import torch

def test_tensor_clamp():
    """
    将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。
    即：小于min (Number) 的当做min (Number) ，大于 max (Number)当做max (Number)
    input (Tensor) – 输入张量
    min (Number) – 限制范围下限
    max (Number) – 限制范围上限
    out (Tensor, optional) – 输出张量
    :return:
    """
    a = torch.randint(low=0, high=10, size=(1, 10))
    print(a)
    a = torch.clamp(a, 3, 5)
    print(a)

if __name__ == '__main__':
    """PyTorch tensors来创建前向神经网络，计算损失，以及反向传播。"""
    N,D_in,H,D_out = 64,1000,100,10
    # 随机创建一些训练数据
    # N 行 D_in 列
    x = torch.randn(N,D_in)# 64 * 1000
    print(x.shape)
    # N 行 D_out 列
    y = torch.randn(N,D_out)# 64*10
    print(y.shape)

    # D_in 行， H 列, 1000 行，100列,表示第一层有 100 个神经元
    w1 = torch.randn(D_in,H)#1000*100
    # H 行， D_out 列, 100 行，10列,表示第二层有 10 个神经元
    w2 = torch.randn(H,D_out)#100*10
    print(w1.shape)
    print(w2.shape)
    learning_rate = 10**-6
    for it in range(500):
        # forward pass
        h = x.mm(w1)#第一层前向传播,64*100
        #     将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。
        #     即：小于min (Number) 的当做min (Number) ，大于 max (Number)当做max (Number)
        h_relu = h.clamp(min=0)# N * H，类似于激活函数 64*100
        # print(h_relu.shape)
        # 矩阵相乘,即 第一层激活后的矩阵和第二次的权重参数相乘,
        # N * D_out ,神经网络第二层的输出
        y_pred = h_relu.mm(w2)#64*10
        # print(y_pred.shape)

        # pow(2)对应位置平方，.sum()矩阵所有元素之和,.item()取出张量中的数值
        loss = (y_pred - y).pow(2).sum().item()
        print(" 第 {} 次 前向传播结束，总体损失为 {:.10f}".format(it,loss))

        # Backward pass
        # compute the gradient
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()
        grad_h[h<0] = 0
        grad_w1 = x.t().mm(grad_h)

        # 更新参数
        w1 = w1 - learning_rate * grad_w1
        w2 = w2 - learning_rate * grad_w2




