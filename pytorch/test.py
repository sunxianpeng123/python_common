# encoding: utf-8

"""
@author: sunxianpeng
@file: 1_indoor.py
@time: 2019/11/3 18:31
"""
import torch
import torchvision
if __name__ == '__main__':
    torch.empty(5,3)
    torch.zeros(5,3)
    x = torch.rand(5,3)

    print(torch.__version__)
    print(torchvision.__version__)
