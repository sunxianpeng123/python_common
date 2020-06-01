# -*- coding: utf-8 -*-
# @Time : 2019/12/28 0:56
# @Author : sxp
# @Email : 
# @File : 3_matrix_compute.py
# @Project : python_common

import tensorflow as tf


def tf_matrix_2_dims():
    """ 2维 矩阵的计算,必须为浮点型数据"""
    print("#############1、tf_matrix_2_dims ################")
    a = tf.fill([2,3],1.)
    b = tf.random.normal([3,5])

    # 下面两种方式都是进行矩阵的相乘
    t_1 = a @ b
    t_2 = tf.matmul(a, b)

    print('t_1 shape = {}'.format(t_1.shape))
    print('t_2 shape = {}'.format(t_2.shape))
    return None

def tf_martix_multi_dims():
    """ 多维矩阵的计算"""
    print("#############2、tf_martix_multi_dims ################")
    a = tf.ones([4, 2, 3])
    b = tf.fill([4, 3, 5], 2.)

    # 下面两种方式都是进行矩阵的相乘,并且表示相同的意思
    # 在这个例子中，将第一维度4代表batch，后面两个维度作为矩阵相乘，
    # 即 最终结果为 4 * [（2*3) * (3*5）]
    t_1 = a @ b
    t_2 = tf.matmul(a,b)

    print('t_1 shape = {}'.format(t_1.shape))
    print('t_2 shape = {}'.format(t_2.shape))
    # t_1 shape = (4, 2, 5)
    # t_2 shape = (4, 2, 5)
    return None

def tf_matrix_with_broadcasting():
    """ 广播 """
    print("#############3、tf_matrix_with_broadcasting ################")
    a = tf.ones([4, 2, 3])
    b = tf.fill([3, 5],2.)
    bb = tf.broadcast_to(b,[4,3,5])

    t_1 = tf.matmul(a,bb)#计算结果和tf_martix_multi_dims中相同
    print('t_1 shape = {}'.format(t_1.shape))
    # t_1 shape = (4, 2, 5)
    return None


if __name__ == '__main__':
    tf_matrix_2_dims()
    tf_martix_multi_dims()
    tf_matrix_with_broadcasting()