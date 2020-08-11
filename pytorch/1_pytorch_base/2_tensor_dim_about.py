# encoding: utf-8

"""
@author: sunxianpeng
@file: tensor_operation.py
@time: 2019/11/24 15:30
"""

import torch

"""查看tensor的整体属性"""


def check_tensor_attr():
    float_tensor = torch.ones(2, 2)
    print(float_tensor.type())
    # 转为整形
    long_tensor = float_tensor.long()
    """ 操作"""
    x = torch.randn(4, 3)
    print(x)
    print("x.shape = {}".format(x.shape))
    print("############## 1、在行上取最大值和最大值下标 ##############")
    # x.shape = torch.Size([4, 3]) ,dim = 1表示将数据进行分组，每组个数为第二个维度上的大小
    max_value, max_idx = torch.max(x, dim=1)
    print("max value ={}\nmax index = {}".format(max_value, max_idx))
    print("############## 2、求每行或每列的和 ##############")
    sum_x = torch.sum(x, dim=1)
    sum_y = torch.sum(x, dim=0)
    print("sum x ={}\nsum y = {}".format(sum_x, sum_y))


""" 增加和减少维度"""


def change_tensor_dim_num():
    float_tensor = torch.ones(2, 2)
    print(float_tensor.type())
    # 转为整形
    long_tensor = float_tensor.long()
    """ 操作"""
    x = torch.randn(4, 3)
    print(x)
    print("############## 3、增加维度或者减少维度 ##############")
    print("==========（1）增加维度")
    print("x shape = {}".format(x.shape))  # torch.Size([4, 3])
    x_1 = x.unsqueeze(0)  # 在第一维度增加
    print("x_1 shape ={}".format(x_1.shape))  # torch.Size([1, 4, 3])
    x_2 = x_1.unsqueeze(1)  # 在第二维度增加
    print("x_2 shape ={}".format(x_2.shape))  # torch.Size([1, 1, 4, 3])
    print("==========（2）减少维度")
    x_3 = x_2.squeeze(0)  # 减少第一维
    print("x_3 shape = {}".format(x_3.shape))  # torch.Size([1, 4, 3])
    x_4 = x_3.squeeze()  # 去掉所有一维的维度
    print("x_4 shape = {}".format(x_4.shape))  # torch.Size([4, 3])
    x_5 = x_2.squeeze()  # 去掉所有一维的维度，torch.Size([1, 1, 4, 3])
    print("x_5 shape = {}".format(x_5.shape))  # torch.Size([4, 3])


"""重排维度顺序、维度变换"""

def change_tensor_dim_order():
    print("############## 4、重新排列维度的顺序 ##############")
    x = torch.randn(3, 4, 5)  # torch.Size([3, 4, 5])
    print("x shape = {}".format(x.shape))
    # 重新排列维度顺序，序号代表原先维度的下标
    x_1 = x.permute(1, 0, 2)
    print("x_1 shape = {}".format(x_1.shape))  # torch.Size([4, 3, 5])
    # 交换指定下标的两个维度顺序
    x_2 = x.transpose(0, 2)
    print("x_2 shape = {}".format(x_2.shape))  # torch.Size([5, 4, 3])
    print("############## 5、reshape 原tensor形状 ##############")
    #  -1 表示任意大小， 5 表示将第二维度变成 5
    x_3 = x.view(-1, 5)
    print("x_3 shape = {}".format(x_3.shape))  # torch.Size([12, 5])
    # 将tensor重新reshape成（3,20）大小
    x_4 = x.view(3, 20)
    print("x_4 shape = {}".format(x_4.shape))  # torch.Size([3, 20])
    print("############## 6、直接在原tensor操作，不单独开辟空间 ##############")
    ones = torch.ones(3, 3)
    twos = torch.ones(3, 3)
    print("one shape ={}".format(ones.shape))  # torch.Size([3, 3])
    ones.unsqueeze_(0)
    print("one shape ={}".format(ones.shape))  # torch.Size([1, 3, 3])
    ones.transpose_(1, 0)
    print("one shape ={}".format(ones.shape))  # torch.Size([3, 1, 3])
    print("=========计算两个tensor之和")
    ones = torch.ones(3, 3)
    twos = torch.ones(3, 3)
    ones.add_(twos)
    print(ones)


def exercise_1():
    print("################练习##################")
    x = torch.ones(4, 4).float()
    x[1:3, 1:3] = 2
    print(x)


if __name__ == '__main__':
    check_tensor_attr()
    change_tensor_dim_num()
    change_tensor_dim_order()
    exercise_1()
