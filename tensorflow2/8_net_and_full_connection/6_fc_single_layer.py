# encoding: utf-8

"""
@author: sunxianpeng
@file: 6_full_connection.py
@time: 2020/1/2 13:34
"""


import tensorflow as tf
from keras.layers import Dense

def tf_Dense():
    """
    tf.layers.dense(
    inputs,					#层的输入
    units,					#该层的输出维度
    activation=None,		#激活函数
    use_bias=True,
    kernel_initializer=None,  	# 卷积核的初始化器
    bias_initializer=tf.zeros_initializer(),  # 偏置项的初始化器
    kernel_regularizer=None,    # 卷积核的正则化
    bias_regularizer=None,    	# 偏置项的正则化
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,  # 层的名字
    reuse=None  # 是否重复使用参数
    )

    """
    print("########################1、tf_Dense #######################")
    x = tf.random.normal([4, 784])
    # 512 指的是前一层的输出维度，会自动生成 w 、b参数
    net = Dense(512)
    out = net(x)

    print('out shape = {}'.format(out.shape))
    # w 矩阵的形状
    print('net keral shape = {}'.format(net.kernel.shape))
    # 偏执量 b 的形状
    print('net bias shape = {}'.format(net.bias.shape))

    return None

def tf_build():
    print("########################2、tf_build #######################")
    net = Dense(10)
    # 在声明 Dense的时候并没有创建对应参数，所有以下会报错，或者返回空值
    # AttributeError: 'Dense' object has no attribute 'bias'
    # print(net.bias)
    # print(net.get_weights())#[]
    # print(net.weights)#[]
    """多次使用build"""
    # 多次build会改变 w 、b 等参数的形状
    # 输入维度 4
    net.build(input_shape=(None,4))
    # w 矩阵的形状
    print('net keral shape = {}'.format(net.kernel.shape))
    # 偏执量 b 的形状
    print('net bias shape = {}'.format(net.bias.shape))
    print("=============")
    net.build(input_shape=(None,20))
    # w 矩阵的形状
    print('net keral shape = {}'.format(net.kernel.shape))
    # 偏执量 b 的形状
    print('net bias shape = {}'.format(net.bias.shape))
    print("=============")
    net.build(input_shape=(2,4))
    # w 矩阵的形状
    print('net keral shape = {}'.format(net.kernel.shape))
    # 偏执量 b 的形状
    print('net bias shape = {}'.format(net.bias.shape))

    return None

def tf_input():
    print("########################3、tf_input #######################")
    net = Dense(10)
    # 设置输入维度为20
    net.build(input_shape=(None,20))
    # ValueError: Input 0 is incompatible with layer dense_3: expected axis -1 of input shape to have value 20 but got shape (2, 12)
    # 单数数据的维度为12，所有会报错
    # input_data = tf.random.normal((4,12))#error
    input_data = tf.random.normal((4,20))#true

    out = net(input_data)

    print('out shape = {}'.format(out.shape))

    return None


if __name__ == '__main__':
    tf_Dense()
    tf_build()
    tf_input()