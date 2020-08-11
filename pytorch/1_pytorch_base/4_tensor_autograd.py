# encoding: utf-8

"""
@author: sunxianpeng
@file: 4_tensor_autograd.py
@time: 2019/11/24 18:43
"""
import torch


def tensor_autograd_demo_1():
    x = torch.tensor(1., requires_grad=True)
    w = torch.tensor(2., requires_grad=True)
    b = torch.tensor(3., requires_grad=True)

    y = w * x + b  # y = 2*1+3
    y.backward()

    # dy / dw = x
    print(w.grad)
    print(x.grad)
    print(b.grad)


def tensor_autograd_demo_2():
    N, D_in, H, D_out = 64, 1000, 100, 10

    # 随机创建一些训练数据
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

    w1 = torch.randn(D_in, H, requires_grad=True)
    w2 = torch.randn(H, D_out, requires_grad=True)

    learning_rate = 1e-6
    for it in range(500):
        # Forward pass
        y_pred = x.mm(w1).clamp(min=0).mm(w2)

        # compute loss
        loss = (y_pred - y).pow(2).sum()  # computation graph
        print(it, loss.item())

        # Backward pass
        loss.backward()

        # update weights of w1 and w2
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad
            w1.grad.zero_()
            w2.grad.zero_()


if __name__ == '__main__':
    tensor_autograd_demo_1()
    tensor_autograd_demo_2()
