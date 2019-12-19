# -*- coding: utf-8 -*-
# @Time : 2019/12/20 1:47
# @Author : sxp
# @Email : 
# @File : 1_func_api.py
# @Project : python_common

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

"""
    tf.keras.Sequential 模型是层的简单堆叠，无法表示任意模型。使用 Keras 函数式 API 可以构建复杂的模型拓扑，例如：
    多输入模型，
    多输出模型，
    
    具有共享层的模型（同一层被调用多次），
    
    具有非序列数据流的模型（例如，残差连接）。
    
    使用函数式 API 构建的模型具有以下特征：
    
    层实例可调用并返回张量。 输入张量和输出张量用于定义 tf.keras.Model 实例。 此模型的训练方式和 Sequential 模型一样。
"""


if __name__ == '__main__':
    """data"""
    train_x = np.random.random((1000, 72))
    train_y = np.random.random((1000, 10))
    val_x = np.random.random((200, 72))
    val_y = np.random.random((200, 10))
    """model"""
    input_x = tf.keras.Input(shape=(72,))
    hidden1 = layers.Dense(32, activation='relu')(input_x)
    hidden2 = layers.Dense(16, activation='relu')(hidden1)
    pred = layers.Dense(10, activation='softmax')(hidden2)
    """other"""
    model = tf.keras.Model(inputs=input_x, outputs=pred)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    model.fit(train_x, train_y, batch_size=32, epochs=5)