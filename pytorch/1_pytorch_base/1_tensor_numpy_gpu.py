# encoding: utf-8

"""
@author: sunxianpeng
@file: tensor_about.py
@time: 2019/11/24 14:34
"""
import torch
import numpy as np


def tensor_numpy():
    array = np.random.randn(10, 20)
    print("############### 一、tensor 和 numpy的互相转化#################")
    print("*****1、 numpy 转为 tensor")
    tensor_1 = torch.from_numpy(array)
    tensor_2 = torch.Tensor(array)
    print("tensor_1 .shape = {}\ntensor_2 shape = {}".format(tensor_1.shape, tensor_2.shape))
    print()
    print("*******2、tensor 转为 numpy")
    # 1、tensor是在cpu上运行时
    array_1 = tensor_1.numpy()
    # 2、tensor是在gpu运行时
    array_2 = tensor_2.cpu().numpy()


def use_gpu():
    print("###############二、使用 GPU 加速 #################")
    print("*********1、tensor放在gpu")
    print("=====（1）定义cuda数据类型")
    gpu_tensor_1 = torch.randn(10, 20, dtype=torch.float)
    gpu_tensor_2 = torch.randn(10, 20).float()
    gpu_tensor_3 = torch.randn(10, 20).cuda().float().cpu().float()

    print("====（2）将不同的tensor放在不停的gpu上")
    gpu_tensor_4 = torch.randn(10, 20).cuda(0)
    gpu_tensor_5 = torch.randn(10, 20).cuda(1)
    print("======(3)使用to方法转换cpu或者gpu")
    # 使用.to方法，Tensor可以被移动到别的device上。
    print('aaa', True if torch.cuda.is_available() else False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.ones(3, 3).to(device=device)
    y = torch.ones(3, 3, device=device)
    print("*********2、将gpu上的tensor放回cpu")
    cpu_tensor = gpu_tensor_1.cpu()
    cpu_tensor_1 = gpu_tensor_1.to('cpu')


def check_tensor_attr():
    print("###################三、访问 tensor的属性########################")
    tensor = torch.randn(10, 20)
    # tensor dim tensor维度
    # tensor numel 元素个数
    print("tensor shape = {}\ntensor size = {}\ntensor type = {}\ntensor dim = {}\ntensor numel = {}"
          .format(tensor.shape, tensor.size(), tensor.type(), tensor.dim(), tensor.numel()))


if __name__ == '__main__':
    tensor_numpy()
    # use_gpu()
    check_tensor_attr()
