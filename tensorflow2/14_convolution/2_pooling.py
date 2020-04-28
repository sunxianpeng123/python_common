# -*- coding: utf-8 -*-
# @Time : 2020/1/10 1:19
# @Author : sxp
# @Email : 
# @File : 2_pooling.py
# @Project : python_common

import tensorflow as tf
from tensorflow.keras import layers


def tf_pooling():
    """下采样，缩小"""
    print("#####################1、tf_pooling ###################")
    x = tf.random.normal([1, 32, 32, 3])# 1 张 32*32*3 的图像
    # 大小为 2*2 的核，步长为2
    #  W = ( W - F ) / S + 1 = (32 - 2) / 2 + 1 = 16
    #  H = ( H - F ) / S + 1 = (32 - 2) / 2 + 1 = 16
    pool = layers.MaxPool2D(2, strides=2)
    out = pool(x)
    print("out.shape = {}".format(out.shape))
    print("==============")
    # 舍掉小数，取整数
    #  W = ( W - F ) / S + 1 = (32 - 3) / 2 + 1 = 15
    #  H = ( H - F ) / S + 1 = (32 - 3) / 2 + 1 = 15
    pool_1 = layers.MaxPool2D(3, strides=2)
    out_1 = pool_1(x)
    print("out_1.shape = {}".format(out_1.shape))
    print("==============")
    #  W = ( W - F ) / S + 1 = (32 - 2) / 2 + 1 = 16
    #  H = ( H - F ) / S + 1 = (32 - 2) / 2 + 1 = 16
    out_2 = tf.nn.max_pool2d(x, 2,strides=2,padding='VALID')
    print("out_2.shape = {}".format(out_2.shape))
    return None

def tf_upsample():
    """上采样，放大"""
    print("#####################2、tf_upsample ###################")
    x = tf.random.normal([1, 7, 7, 4])# 1 张 7*7*4 的图像
    # 沿着数据的行和列分别重复size次
    layer = layers.UpSampling2D(size=3)
    out = layer(x)
    print("out.shape = {}".format(out.shape))
    # (1, 21, 21, 4)
    print("==============")
    layer = layers.UpSampling2D(size=2)
    out = layer(x)
    print("out.shape = {}".format(out.shape))
    # (1, 14, 14, 4)
    return None


if __name__ == '__main__':
    tf_pooling()
    tf_upsample()