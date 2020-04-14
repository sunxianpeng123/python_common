# encoding: utf-8

"""
@author: sunxianpeng
@file: 6_create_tensor_zeros.py
@time: 2019/12/23 19:46
"""
import tensorflow as tf
import numpy as np

def tf_constant():
    """constant常量"""
    print("##############1、constant###################")
    # 均值和方差均为 1
    t_1 = tf.constant(1)
    # 均值为 0 方差为 1
    t_2 = tf.constant([1])
    t_3 = tf.constant([1,2.])
    t_4 = tf.constant([[1,2.],[3.,4.]])
    print(t_4)

    print(t_1)
    print(t_1.dtype)
    print('t_1 shape = {}'.format(t_1.shape))

    print(t_2)
    print(t_2.dtype)
    print('t_2 shape = {}'.format(t_2.shape))

    print(t_3)
    print(t_3.dtype)
    print('t_3 shape = {}'.format(t_3.shape))
    # <dtype: 'int32'>
    # t_1 shape = ()
    # <dtype: 'int32'>
    # t_2 shape = (1,)
    # <dtype: 'float32'>
    # t_3 shape = (2,)

    return None


if __name__ == '__main__':
    tf_constant()