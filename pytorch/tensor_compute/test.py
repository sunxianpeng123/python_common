# encoding: utf-8

import torch
import numpy as np

def test_torch_all():
    # 是否存在为0的元素
    r1 = torch.all(torch.ByteTensor([1, 1, 1, 1]))
    r2 = torch.all(torch.ByteTensor([1, 1, 1, 0]))
    print("r1 = {}".format(r1))
    print("r2 = {}".format(r2))
    # r1 = 1
    # r2 = 0



if __name__ == '__main__':
    test_torch_all()


