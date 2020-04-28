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
    # 图像尺寸（W*H）
    # 卷积后输出的图像 w = (W - kernel_size + 2 * 0 ) / strides + 1 = 23 / 1 +1 =24
    # 卷积后输出的图像 h=(H - kernel_size + 2 * 0) / strides + 1 = 23 / 1 + 1 =24
    # 4 表示使用4个卷积核，每个卷积核的shape为（3， 5，5）即表示三个channel的5*5大小的核
    layer1 = layers.Conv2D(4, kernel_size=[5,5], strides=[1,1], padding='valid')
    layer2 = layers.Conv2D(4, kernel_size=5, strides=1, padding='valid')
    out_1 = layer1(x)
    out_2 = layer1.call(x)
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
    # 第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，具
    #                  体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，要求类型为float32和float64其中之一
    # 第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
    #                   具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
    # 第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，
    # 第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，设置为SAME，则说明输入图片大小和输出图片大小是一致的，如果是VALID则图片经过滤波器后可能会变小。
    # 第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true
    # 结果返回一个Tensor，这个输出，就是我们常说的feature map，shape仍然是[batch, height, width, channels]这种形式。
    out = tf.nn.conv2d(x, filters=w, strides=1, padding='VALID')
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
