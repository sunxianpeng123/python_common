# -*- coding: utf-8 -*-
# @Time : 2019/12/20 1:51
# @Author : sxp
# @Email : 
# @File : 2_self_defined_model.py
# @Project : python_common

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
"""
    通过对 tf.keras.Model 进行子类化并定义您自己的前向传播来构建完全可自定义的模型。
    在 init 方法中创建层并将它们设置为类实例的属性。在 call 方法中定义前向传播
"""

# 继承
class MyModel(tf.keras.Model):
    # 调用父类中的init方法
    def __init__(self,num_classes):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        self.layer1 = layers.Dense(32,activation='relu')
        self.layer2 = layers.Dense(num_classes,activation='softmax')
    # 构建网络
    def call(self,inputs):
        h1 = self.layer1(inputs)
        out = self.layer2(h1)
        return out
    def compute_out_put_shape(self,input_shape):
        # 表示tensor的shape,获取每一层网络的输出的维度?并转成list
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)


if __name__ == '__main__':
    train_x = np.random.random((1000,72))
    train_y = np.random.random((1000,10))
    val_x = np.random.random((200,72))
    val_y = np.random.random((200,10))

    model = MyModel(num_classes=10)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    model.fit(train_x, train_y, batch_size=16, epochs=5)
