# encoding: utf-8

"""
@author: sunxianpeng
@file: 6_create_tensor_zeros.py
@time: 2019/12/23 19:46
"""
import tensorflow as tf
import numpy as np

def from_numpy_list():
    "从 numpy,list 构建tensor"
    print("##########1、从 numpy,list 构建tensor#############")
    ones = np.ones([2,3])
    zeros = np.zeros([2,3])
    list_1 = [1,2]
    list_2 = [1,2.]
    list_3 = [[1,2],[3,4.]]

    t_1 = tf.convert_to_tensor(ones)
    t_2 = tf.convert_to_tensor(zeros)
    t_3 = tf.convert_to_tensor(list_1)
    t_4 = tf.convert_to_tensor(list_2)
    t_5 = tf.convert_to_tensor(list_3)

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
    # <dtype: 'float64'>
    # t_1 shape = (2, 3)
    # t_1 = [[1. 1. 1.]
    #  [1. 1. 1.]]
    # <dtype: 'float64'>
    # t_2 shape = (2, 3)
    # t_2 = [[0. 0. 0.]
    #  [0. 0. 0.]]
    # <dtype: 'int32'>
    # t_3 shape = (2,)
    # t_3 = [1 2]
    # <dtype: 'float32'>
    # t_4 shape = (2,)
    # t_4 = [1. 2.]
    # <dtype: 'float32'>
    # t_5 shape = (2, 2)
    # t_5 = [[1. 2.]
    #  [3. 4.]]

    return None

if __name__ == '__main__':
    from_numpy_list()
