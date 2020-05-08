# encoding: utf-8

"""
@author: sunxianpeng
@file: 1_helloworld_1.py
@time: 2019/11/6 17:45
"""


import numpy as np
import torch
import torch.nn as nn

def auto_grad():
    # 一个PyTorchTensor很像一个numpy的ndarray。但是它和numpyndarray最大的区别是，PyTorch
    # Tensor可以在CPU或者GPU上运算。如果想要在GPU上运算，就需要把Tensor换成cuda类型。
    x = torch.tensor(1,requires_grad=True)
    w = torch.tensor(2,requires_grad=True)
    b = torch.tensor(3,requires_grad=True)
    y = w * x + b # y = 2 * 1 + 3
    # dy/dw = x
    print(w.grad)
    print(x.grad)
    print(b.grad)

if __name__ == '__main__':
    """使用PyTorch中nn这个库来构建网络。 用PyTorch autograd来构建
    计算图和计算gradients， 然后PyTorch会帮我们自动计算gradient。"""
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
    learning_rate = 10**-6

    for it in range(500):
        # forward pass
        y_pred = model(x)
        # compute loss
        loss = loss_fn(y_pred,y)
        print(" 第 {} 次 前向传播结束，总体损失为 {:.10f}".format(it,loss.item()))
        # Backward pass
        loss.backward()
        # update weights of w1 and w2
        # 使用torch.no_grad()方法就不会再记下weights of w1 and w2，减少内存占用
        with torch.no_grad():
            # param (tensor, grad)
            for param in model.parameters():
                param -= learning_rate * param.grad
        # 重置梯度
        model.zero_grad()






