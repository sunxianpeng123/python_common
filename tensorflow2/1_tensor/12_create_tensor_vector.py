# encoding: utf-8

"""
@author: sunxianpeng
@file: 6_create_tensor_zeros.py
@time: 2019/12/23 19:46
"""
import tensorflow as tf
import numpy as np

def tf_vector():
    """向量，w * x + b """
    print("##############1、tf_vector###################")
    net = tf.keras.layers.Dense(10)# 输入8维,输出10维，4*8 * kernel= 4 * 10
    net.build((4,8))#4*8
    # w
    print(net.kernel)# shape=(8, 10) dtype=float32
    # 1维，10个
    print(net.bias)#shape=(10,) dtype=float32

    return None


if __name__ == '__main__':
    tf_vector()