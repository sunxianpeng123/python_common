# encoding: utf-8

"""
@author: sunxianpeng
@file: 6_create_tensor_zeros.py
@time: 2019/12/23 19:46
"""
import tensorflow as tf
import numpy as np
import keras
from keras import layers

def tf_matrix():
    """向量，w * x + b """
    print("##############1、tf_vector###################")
    x = tf.random.normal([4,784])

    net = layers.Dense(10)
    # w
    net.build((4,784))
    # w.T * x
    print('net          print(dir)(x).shape = {}'.format(net(x).shape))
    print('net kernel shape = {}'.format(net.kernel.shape))
    print('net bias shape = {}'.format(net.bias.shape))
    # net(x).shape = (4, 10)
    # net kernel shape = (784, 10)
    # net bias shape = (10,)

    return None


if __name__ == '__main__':
    tf_matrix()