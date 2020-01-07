# encoding: utf-8

"""
@author: sunxianpeng
@file: 4_self_define_layer.py
@time: 2020/1/7 11:15
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.keras.backend.clear_session()  # For easy reset of notebook state.
from tensorflow.keras import layers


class Linear_1(layers.Layer):
    def __init__(self,input_dim=32, output_dim=32):
        """
        w 和 b 设置为layer的属性，会被层自动跟踪
        :param input_dim:
        :param output_dim:
        """
        super(Linear_1, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim,output_dim)),
                             dtype=tf.float32,trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(output_dim,)),
                             dtype=tf.float32,trainable=True)
    def call(self,inputs):
        return tf.matmul(inputs, self.w) + self.b
##################################################################
##################################################################
class Linear_2(layers.Layer):
    def __init__(self,input_dim=32, output_dim=32):
        super(Linear_2, self).__init__()
        self.w = self.add_weight(shape=(input_dim, output_dim),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(output_dim,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self,inputs):
        return tf.matmul(inputs, self.w) + self.b

##################################################################
#以上添加的 wb 都是可训练的，layer也可以添加不可训练的层
##################################################################

if __name__ == '__main__':
    x = tf.ones((2, 2))
    # linear_layer = Linear_1(2,4)
    linear_layer = Linear_2(2,4)
    y = linear_layer(x)
    print(y)