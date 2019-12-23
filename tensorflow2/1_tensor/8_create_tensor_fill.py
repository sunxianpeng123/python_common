# encoding: utf-8

"""
@author: sunxianpeng
@file: 6_create_tensor_zeros.py
@time: 2019/12/23 19:46
"""
import tensorflow as tf
import numpy as np

def tf_fill():
    """使用指定值填充tensor"""
    t_1 = tf.fill([2,2],0)
    t_2 = tf.fill([2,2],0.)
    t_3 = tf.fill([2,2],1)
    t_4 = tf.fill([2,2],9)

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
    # <dtype: 'int32'>
    # t_1 shape = (2, 2)
    # t_1 = [[0 0]
    #  [0 0]]
    # <dtype: 'float32'>
    # t_2 shape = (2, 2)
    # t_2 = [[0. 0.]
    #  [0. 0.]]
    # <dtype: 'int32'>
    # t_3 shape = (2, 2)
    # t_3 = [[1 1]
    #  [1 1]]
    # <dtype: 'int32'>
    # t_4 shape = (2, 2)
    # t_4 = [[9 9]
    #  [9 9]]
    return None


if __name__ == '__main__':
    tf_fill()