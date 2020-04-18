# encoding: utf-8

"""
@author: sunxianpeng
@file: 6_create_tensor_zeros.py
@time: 2019/12/23 19:46
"""
import tensorflow as tf
import numpy as np

def tf_ones_and_ones_like():
    """创建全为 1 的tensor"""
    print("##########2、从 tf_ones_and_ones_like 构建tensor#############")
    t_1 = tf.ones(1) # <==> tf.ones([1])
    t_2 = tf.ones([])
    t_3 = tf.ones([2])
    t_4 = tf.ones([2,3])
    t_5 = tf.ones_like(t_4)

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
    print(t_5.dtype)
    print('t_5 shape = {}'.format(t_5.shape))
    print('t_5 = {}'.format(t_5))
    # <dtype: 'float32'>
    # t_1 shape = (1,)
    # t_1 = [1.]
    # <dtype: 'float32'>
    # t_2 shape = ()
    # t_2 = 1.0
    # <dtype: 'float32'>
    # t_3 shape = (2,)
    # t_3 = [1. 1.]
    # <dtype: 'float32'>
    # t_4 shape = (2, 3)
    # t_4 = [[1. 1. 1.]
    #  [1. 1. 1.]]
    # <dtype: 'float32'>
    # t_5 shape = (2, 3)
    # t_5 = [[1. 1. 1.]
    #  [1. 1. 1.]]
    return None


if __name__ == '__main__':
    tf_ones_and_ones_like()