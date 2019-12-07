# encoding: utf-8

"""
@author: sunxianpeng
@file: test.py
@time: 2019/11/7 14:37
"""
import numpy as np
import torch

def fizz_buzz_encode(i):
    if i % 15 ==0: return 3
    elif i % 5 ==0: return 2
    elif i % 3 ==0: return 1
    else:           return 0

def fizz_buzz_decode(i,prediction):
    return [str(i),"fizz","buzz","fizzbuzz"][prediction]


def binary_encode(i,num_digits):
    # 将数字进行二进制反序列表示,切维度为 num_digits
    return np.array([i >> d & 1 for d in range(num_digits)])

if __name__ == '__main__':
    num_digits = 10
    testX = torch.Tensor([binary_encode(i, num_digits) for i in range(1, 101)])
    print(testX)