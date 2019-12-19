# -*- coding: utf-8 -*-
# @Time : 2019/12/20 2:32
# @Author : sxp
# @Email : 
# @File : 1_save_load_weights.py
# @Project : python_common


import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
"""
    这种方法不会保存整个网络的结构，只是保存模型的权重和偏置，所以在后期恢复模型之前，
    必须手动创建和之前模型一模一样的模型，以保证权重和偏置的维度和保存之前的相同。
"""

def train_flow():
    """3.1设置训练流程"""
    model = tf.keras.Sequential()
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=[tf.keras.metrics.categorical_accuracy])
    return model

if __name__ == '__main__':
    train_x = np.random.random((1000,72))
    train_y = np.random.random((1000,10))

    val_x = np.random.random((200,72))
    val_y = np.random.random((200,10))

    model = train_flow()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    model.fit(train_x,train_y,epochs=10,batch_size=100,validation_data=(val_x,val_y))


    print(model.weights)
    model.save_weights('./weights/model')
    model.load_weights('./weights/model')
    model.save_weights('./model.h5')
    model.load_weights('./model.h5')