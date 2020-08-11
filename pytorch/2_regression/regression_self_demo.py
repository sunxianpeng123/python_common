# encoding: utf-8

import torch
import numpy as np

"多预测值线性回归"


def get_data():
    inputs = np.array([[73, 67, 43],
                       [91, 88, 64],
                       [87, 134, 58],
                       [102, 43, 37],
                       [69, 96, 70]], dtype='float32')
    targets = np.array([[56, 70],
                        [81, 101],
                        [119, 133],
                        [22, 37],
                        [103, 119]], dtype='float32')
    return inputs, targets


def model(x):
    # pytorch数学运算
    # https://www.cnblogs.com/taosiyu/p/11599157.html
    return x @ w.t() + b


def mse(t1, t2):
    """平方差矩阵"""
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()


if __name__ == '__main__':

    inputs, targets = get_data()
    inputs = torch.from_numpy(inputs)
    targets = torch.from_numpy(targets)
    # inputs shape = torch.Size([5, 3]),targets shape = torch.Size([5, 2])
    print("inputs shape = {},targets shape = {}".format(inputs.shape, targets.shape))
    # 定义变量和截距(偏差)
    w = torch.randn(2, 3, requires_grad=True)
    b = torch.randn(2, requires_grad=True)
    print(w)
    print(b)
    #
    for i in range(100):
        # x * w + b
        preds = model(inputs)
        loss = mse(preds, targets)
        # 计算梯度
        loss.backward()
        # print("w.grad = ", w.grad)
        with torch.no_grad():
            w -= w.grad * (10 ** -5)
            b -= b.grad * (10 ** -5)
            # 重置梯度
            w.grad.zero_()
            b.grad.zero_()

    print("loss = ", loss)
    print("preds = ", preds)
