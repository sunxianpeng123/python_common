# encoding: utf-8

"""
@author: sunxianpeng
@file: 3_tensor_compute.py
@time: 2019/11/24 17:25
"""

import torch


def add():
    x = torch.rand(5, 3)
    y = torch.randn(5, 3)
    print("###########1、tensor 加法################")
    print("==========（1）两种相加方式")
    add_1 = x + y
    add_2 = torch.add(x, y)
    if add_1.equal(add_2):
        print(True)
    print("=========（2）将相加结果作为一个变量")
    add_3 = torch.empty(5, 3)
    torch.add(x, y, out=add_3)
    if add_1.equal(add_3):
        print(True)
    print("==========(3)in-place加法")
    x.add_(y)  # 将 y 加到x 上，相加后x为最终结果
    if add_1.equal(x):
        print(True)


if __name__ == '__main__':
    add()
#     tensor数学运算参见链接：
#     https://blog.csdn.net/s294878304/article/details/102945910#%EF%BC%881%EF%BC%89%E5%8A%A0%E6%B3%95%E8%BF%90%E7%AE%97
