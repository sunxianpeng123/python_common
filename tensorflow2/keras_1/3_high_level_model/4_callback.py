# -*- coding: utf-8 -*-
# @Time : 2019/12/20 2:26
# @Author : sxp
# @Email : 
# @File : 4_callback.py
# @Project : python_common
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
"""
在我们训练神经网络时，对于训练数据的迭代次数（epochs）的设置，是一个值得思考的问题。
通常，epochs 越大，最后训练的损失值会越小，但是迭代次数过大，会导致过拟合的现象。
我们往往希望当loss值，或准确率达到一定值后，就停止训练。但是我们不可能去人为的等待或者控制。
tensorfow 中的回调机制，就为我们很好的处理了这个问题。
tensorfow 中的回调机制，可以实现在每次迭代一轮后，自动调用制指定的函数（例如：on_epoch_end() 顾名思义）,
可以方便我们来控制训练终止的时机。
"""


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('loss') < 0.4):
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True

if __name__ == '__main__':
    callbacks = myCallback()

    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images = training_images / 255.0
    test_images = test_images / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])
