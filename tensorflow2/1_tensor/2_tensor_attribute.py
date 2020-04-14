# encoding: utf-8

"""
@author: sunxianpeng
@file: 2_tensor_attribute.py
@time: 2019/12/22 19:54
"""

import tensorflow as tf

def device_info():
    """设备信息"""
    print("###################1、设备信息######################")
    print("tensor 在gpu、cpu的转换=======================================")
    with tf.device('cpu'):
        a = tf.constant([1])

    with tf.device('cpu'):#在gpu上创建tensor，无gpu时会在cpu创建
        b = tf.range(4)
    # tensor在设备上的转换
    # aa = a.gpu()#无gpu会报错
    bb = tf.identity(b)#b.cpu()

    print('a = {}'.format(a))
    print('a device = {}'.format(a.device))
    print('b = {}'.format(b))
    print('b device = {}'.format(b.device))
    print('bb = {}'.format(bb))
    print('bb device = {}'.format(bb.device))

    # a = [1]
    # a device = /job:localhost/replica:0/task:0/device:CPU:0
    # b = [0 1 2 3]
    # b device = /job:localhost/replica:0/task:0/device:CPU:0
    # bb = [0 1 2 3]
    # bb device = /job:localhost/replica:0/task:0/device:CPU:0
    print("tensor 转numpy=======================================")
    b_np = b.numpy()
    print('b_np = {}'.format(b_np))
    # b_np = [0 1 2 3]
    print("tensor常用属性=======================================")
    # b.ndim和 tf.rank(b)可以查看数据的维度,即特征数量. eg:1.1为0维，[1.1]为1维
    ones = tf.ones([3,4,2])# 3 个 4 行列矩阵
    print('b.ndim = {}'.format(b.ndim))
    print('b.rank = {}'.format(tf.rank(b)))
    print('tf.ones rank = {}'.format(tf.rank(ones)))
    print('b.shape = {}'.format(b.shape))
    # print('b.name = {}'.format(b.name))#b.name tf1的遗留问题，tf2可以省略，另外命名
    # b.ndim = 1
    # b.rank = 1
    # tf.ones rank = 3
    # b.shape = (4,)
    return None





if __name__ == '__main__':
    device_info()