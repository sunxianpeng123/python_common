# -*- coding: utf-8 -*-
# @Time : 2020/1/10 1:03
# @Author : sxp
# @Email : 
# @File : 1_conv2d.py
# @Project : python_common

import tensorflow as tf
from tensorflow.keras import layers

def tf_conv2d():
    print("###############1、tf_conv2d ###############")
    x = tf.random.normal([1, 28, 28, 3])
    # 4 表示使用4个卷积核，每个卷积核的shape为（3， 5，5）即表示三个channel的5*5大小的核
    # padding设置为SAME，则说明输入图片大小和输出图片大小是一致的，如果是VALID则图片经过滤波器后可能会变小。
    layer = layers.Conv2D(4, kernel_size=5, strides=1, padding='valid')
    out_1 = layer(x)
    out_2 = layer.call(x)
    print('out_1 shape = {}'.format(out_1.shape))
    print('out_2 shape = {}'.format(out_2.shape))
    # out_1 shape = (1, 24, 24, 4)
    # out_2 shape = (1, 24, 24, 4)
    return None

def tf_weight_bias():
    print("###############2、tf_weight_bias ###############")
    x = tf.random.normal([1, 28, 28, 3])
    layer = layers.Conv2D(4, kernel_size=5, strides=1, padding='valid')
    out_1 = layer(x)
    print('out_1 shape = {}'.format(out_1.shape))
    # weight
    print(layer.kernel.shape)
    # bias
    print(layer.bias.shape)
    # out_1 shape = (1, 24, 24, 4)
    # (5, 5, 3, 4)
    # (4,)
    return None

def tf_nn_conv2d():
    print("###############3、tf_nn_conv2d ###############")
    x = tf.random.normal([1, 32, 32, 3])
    w =tf.random.normal([5, 5, 3, 4])
    b = tf.zeros([4])

    out = tf.nn.conv2d(x, w, strides=1, padding='VALID')
    out = out + b
    print(out.shape)

    out_1 = tf.nn.conv2d(x, w, strides=2, padding='VALID')
    out_1 = out_1 + b
    print(out_1.shape)
    # (1, 28, 28, 4)
    # (1, 14, 14, 4)

    return None

if __name__ == '__main__':
    tf_conv2d()
    tf_weight_bias()
    tf_nn_conv2d()
