# -*- coding: utf-8 -*-
# @Time : 2019/12/28 0:11
# @Author : sxp
# @Email : 
# @File : 1_add_minus_multi_divide.py
# @Project : python_common

import tensorflow as tf

def tf_compute_1():
    """+-*/ 普通的运算， 都是矩阵的对应位置加减乘除，矩阵形状必须相同"""
    print("################1、普通的加减乘除##################")
    # a = tf.fill([2,2],2.)
    a = tf.random.normal([2,2])
    b = tf.ones([2,2])

    t_1 = a + b
    t_2 = a - b
    t_3 = a * b
    t_4 = a / b

    print(a)
    print('a shape = {}'.format(a.shape))
    print(b)
    print('b shape = {}'.format(b.shape))
    print(t_1)
    print('t_1 shape = {}'.format(t_1.shape))
    print(t_2)
    print('t_2 shape = {}'.format(t_2.shape))
    print(t_3)
    print('t_3 shape = {}'.format(t_3.shape))
    print(t_4)
    print('t_4 shape = {}'.format(t_4.shape))
    # tf.Tensor(
    # [[2. 2.]
    #  [2. 2.]], shape=(2, 2), dtype=float32)
    # a shape = (2, 2)

    # tf.Tensor(
    # [[1. 1.]
    #  [1. 1.]], shape=(2, 2), dtype=float32)
    # b shape = (2, 2)

    # tf.Tensor(
    # [[3. 3.]
    #  [3. 3.]], shape=(2, 2), dtype=float32)
    # t_1 shape = (2, 2)

    # tf.Tensor(
    # [[1. 1.]
    #  [1. 1.]], shape=(2, 2), dtype=float32)
    # t_2 shape = (2, 2)

    # tf.Tensor(
    # [[2. 2.]
    #  [2. 2.]], shape=(2, 2), dtype=float32)
    # t_3 shape = (2, 2)

    # tf.Tensor(
    # [[2. 2.]
    #  [2. 2.]], shape=(2, 2), dtype=float32)
    # t_4 shape = (2, 2)
    return None

def tf_compute_2():
    """ // (下取整) 、% (取余)是矩阵 对应位置 计算，矩阵形状必须相同"""
    print("################2、整除和取余##################")
    # a = tf.fill([2,2],2.)
    a = tf.random.normal([2,2])
    b = tf.ones([2,2])

    t_1 = b // a
    t_2 = b % a
    print(a)
    print('a shape = {}'.format(a.shape))
    print(b)
    print('b shape = {}'.format(b.shape))
    print(t_1)
    print('t_1 shape = {}'.format(t_1.shape))
    print(t_2)
    print('t_2 shape = {}'.format(t_2.shape))
    # tf.Tensor(
    # [[2. 2.]
    #  [2. 2.]], shape=(2, 2), dtype=float32)
    # a shape = (2, 2)

    # tf.Tensor(
    # [[1. 1.]
    #  [1. 1.]], shape=(2, 2), dtype=float32)
    # b shape = (2, 2)

    # tf.Tensor(
    # [[0. 0.]
    #  [0. 0.]], shape=(2, 2), dtype=float32)
    # t_1 shape = (2, 2)

    # tf.Tensor(
    # [[1. 1.]
    #  [1. 1.]], shape=(2, 2), dtype=float32)
    # t_2 shape = (2, 2)

    return None
if __name__ == '__main__':
    tf_compute_1()
    tf_compute_2()