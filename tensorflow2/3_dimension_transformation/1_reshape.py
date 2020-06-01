# -*- coding: utf-8 -*-
# @Time : 2019/12/25 23:59
# @Author : sxp
# @Email : 
# @File : 1_reshape.py
# @Project : python_common

import tensorflow as tf

def tf_reshape():
    """变换tensor的维度信息"""
    print("################1、tf_reshape#################")
    t = tf.random.normal([4, 28, 28, 3])#4张28*28*3的照片
    #相当于将照片的各个通道分别打平成一个向量，但是保留通道信息
    t_1 = tf.reshape(t, [4, 784, 3])
    t_2 = tf.reshape(t, [4, -1, 3])#同 t_1，只能有一个-1，-1表示让系统自动计算此处的数值
    # 相当于将图片的各个通道打平成一个向量，不保留通道信息
    t_3 = tf.reshape(t, [4, 784 * 3])
    t_4 = tf.reshape(t, [4, -1])

    print('t shape = {}, ndim = {}'.format(t.shape,t.ndim))
    print('t_1 shape = {}, ndim = {}'.format(t_1.shape,t_1.ndim))
    print('t_2 shape = {}, ndim = {}'.format(t_2.shape,t_2.ndim))
    print('t_3 shape = {}, ndim = {}'.format(t_3.shape,t_3.ndim))
    print('t_4 shape = {}, ndim = {}'.format(t_4.shape,t_4.ndim))
    # t shape = (4, 28, 28, 3), ndim = 4
    # t_1 shape = (4, 784, 3), ndim = 3
    # t_2 shape = (4, 784, 3), ndim = 3
    # t_3 shape = (4, 2352), ndim = 2
    # t_4 shape = (4, 2352), ndim = 2
    return None

def tf_back_reshape():
    """将reshape后的tensor恢复，但是需要知道reshape前的tensor物理意义"""
    print("################2、tf_back_reshape#################")
    t = tf.random.normal([4,28,28,3])#4张28*28*3的照片
    #相当于将照片的各个通道分别打平成一个向量，但是保留通道信息
    t_1 = tf.reshape(t,[4,784,3])
    t_1_back = tf.reshape(t_1,[4,28,28,3])#将t_1回复成t
    # 相当于将图片的各个通道打平成一个向量，不保留通道信息
    t_2 = tf.reshape(t,[4,-1])
    t_2_back = tf.reshape(t_2,[4,14,56,3])#将t_1重置别的tensor，和t不同

    print('t shape = {}, ndim = {}'.format(t.shape,t.ndim))
    print('t_1 shape = {}, ndim = {}'.format(t_1.shape,t_1.ndim))
    print('t_1_back shape = {}, ndim = {}'.format(t_1_back.shape,t_1_back.ndim))
    print('t_2 shape = {}, ndim = {}'.format(t_2.shape,t_2.ndim))
    print('t_2_back shape = {}, ndim = {}'.format(t_2_back.shape,t_2_back.ndim))
    # t shape = (4, 28, 28, 3), ndim = 4
    # t_1 shape = (4, 784, 3), ndim = 3
    # t_1_back shape = (4, 28, 28, 3), ndim = 4
    # t_2 shape = (4, 2352), ndim = 2
    # t_2_back shape = (4, 14, 56, 3), ndim = 4
    return None

if __name__ == '__main__':
    tf_reshape()
    tf_back_reshape()