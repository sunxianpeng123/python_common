# -*- coding: utf-8 -*-
# @Time : 2019/12/19 23:49
# @Author : sxp
# @Email : 
# @File : 1_indoor.py
# @Project : python_common

import tensorflow as tf
import keras
from keras import layers
############################################################
#常见的神经网络都包含在keras.layer中(最新的tf.keras的版本可能和keras不同)
############################################################
def add_model():
    """1、模型堆叠"""
    model = keras.Sequential()
    model.add(layers.Dense(32,activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

def net_conf():
    """2、网络配置"""
    """
    全连接层
    activation：设置层的激活函数。此参数由内置函数的名称指定，或指定为可调用对象。默认情况下，系统不会应用任何激活函数。

    kernel_initializer 和 bias_initializer：创建层权重（核和偏差）的初始化方案。此参数是一个名称或可调用对象，默认为 "Glorot uniform" 初始化器。
    
    kernel_regularizer 和 bias_regularizer：应用层权重（核和偏差）的正则化方案，例如 L1 或 L2 正则化。默认情况下，系统不会应用正则化函数。
    
    inputs: 输入数据，2维tensor. 
    
    units: 该层的神经单元结点数。 
    
    use_bias: Boolean型，是否使用偏置项.  
    
    bias_initializer: 偏置项的初始化器，默认初始化为0. 
    
    bias_regularizer: 偏置项的正则化，可选. 
    
    activity_regularizer: 输出的正则化函数. 
    
    trainable: Boolean型，表明该层的参数是否参与训练。如果为真则变量加入到图集合中 GraphKeys.TRAINABLE_VARIABLES (see tf.Variable). 
    
    name: 层的名字. 
    
    reuse: Boolean型, 是否重复使用参数.
    """
    layers.Dense(32, activation='sigmoid')
    layers.Dense(32, activation=tf.sigmoid)
    layers.Dense(32, kernel_initializer='orthogonal')
    layers.Dense(32, kernel_initializer=keras.initializers.glorot_normal)
    layers.Dense(32, kernel_regularizer=keras.regularizers.l2(0.01))
    layers.Dense(32, kernel_regularizer=keras.regularizers.l1(0.01))






if __name__ == '__main__':
    print(tf.__version__)
    print(keras.__version__)
    add_model()
    net_conf()