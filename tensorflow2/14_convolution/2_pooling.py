# -*- coding: utf-8 -*-
# @Time : 2020/1/10 1:19
# @Author : sxp
# @Email : 
# @File : 2_pooling.py
# @Project : python_common

import tensorflow as tf
from tensorflow.keras import layers


def tf_pooling():
    x = tf.random.normal([1, 32, 32, 3])
    # 大小为2*2的核，步长为2
    # (32 - 2 ) / 2 + 1 = 16
    pool = layers.MaxPool2D(2, strides=2)
    out = pool(x)
    print(out.shape)
    print("==============")
    # (32 - 3) / 2 + 1 = 15
    pool_1 = layers.MaxPool2D(3, strides=2)
    out_1 = pool_1(x)
    print(out_1.shape)
    print("==============")
    #
    out_2 = tf.nn.max_pool2d(x, 2,strides=2,padding='VALID')
    print(out_2.shape)

    return None


if __name__ == '__main__':
    tf_pooling()