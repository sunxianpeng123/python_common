# encoding: utf-8

"""
@author: sunxianpeng
@file: 6_create_tensor_zeros.py
@time: 2019/12/23 19:46
"""
import tensorflow as tf
import numpy as np

def tf_zeros():
    """创建全为0的tensor"""
    print("##########2、从 tf_zeros 构建tensor#############")
    t_1 = tf.zeros([])
    t_2 = tf.zeros([1])
    t_3 = tf.zeros([2,2])
    t_4 = tf.zeros([2,3,3])

    print(t_1.dtype)
    print('t_1 shape = {}'.format(t_1.shape))
    print('t_1 = {}'.format(t_1))
    print(t_2.dtype)
    print('t_2 shape = {}'.format(t_2.shape))
    print('t_2 = {}'.format(t_2))
    print(t_3.dtype)
    print('t_3 shape = {}'.format(t_3.shape))
    print('t_3 = {}'.format(t_3))
    print(t_4.dtype)
    print('t_4 shape = {}'.format(t_4.shape))
    print('t_4 = {}'.format(t_4))
    # <dtype: 'float32'>
    # t_1 shape = ()
    # t_1 = 0.0
    # <dtype: 'float32'>
    # t_2 shape = (1,)
    # t_2 = [0.]
    # <dtype: 'float32'>
    # t_3 shape = (2, 2)
    # t_3 = [[0. 0.]
    #  [0. 0.]]
    # <dtype: 'float32'>
    # t_4 shape = (2, 3, 3)
    # t_4 = [[[0. 0. 0.]
    #   [0. 0. 0.]
    #   [0. 0. 0.]]
    #
    #  [[0. 0. 0.]
    #   [0. 0. 0.]
    #   [0. 0. 0.]]]
    return None

def tf_zero_like():
    """创建和其他向量相同的tensor """
    print("###############3、创建和其他向量相同的tensor####################")
    t_1 = tf.zeros([2,3,3])
    t_2 = tf.zeros_like(t_1)
    t_3 = tf.zeros(t_1.shape)

    print('t_1 shape = {}'.format(t_1.shape))
    print('t_2 shape = {}'.format(t_2.shape))
    print('t_3 shape = {}'.format(t_3.shape))
    # t_1 shape = (2, 3, 3)
    # t_2 shape = (2, 3, 3)
    # t_3 shape = (2, 3, 3)
    return None

if __name__ == '__main__':
    tf_zeros()
    tf_zero_like()