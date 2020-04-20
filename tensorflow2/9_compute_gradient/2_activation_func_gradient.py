# encoding: utf-8

"""
@author: sunxianpeng
@file: 2_activation_func_gradient.py
@time: 2020/1/4 15:51
"""

import tensorflow as tf


def tf_sigmoid_gradient():
    """ sigmoid 导数
    f(x) = 1 / (1 + e^x)
    """
    print("####################1、tf_sigmoid_gradient ####################")
    a = tf.linspace(-10., 10, 10)

    with tf.GradientTape() as tape:
        tape.watch(a)
        y = tf.sigmoid(a)
    # 求解a中每个点的梯度
    grad = tape.gradient(y, [a])
    print('grad = {}'.format(grad))
    return None

def tf_tanh_gradient():
    """ tanh 导数
    f(x) = 2 * sigmoid(2x) - 1
    """
    print("####################2、tf_tanh_gradient ####################")
    a = tf.linspace(-5., 5, 10)
    with tf.GradientTape() as tape:
        tape.watch(a)
        y = tf.tanh(a)

    # 求解a中每个点的梯度
    grad = tape.gradient(y, [a])
    print('grad = {}'.format(grad))
    return None

def tf_relu_gradient():
    """
    一、relu
        f(x) =
              0  x < 0
            x  x >= 0
    二、leaky_relu
        f(x) =

    """
    print("####################3、tf_relu_gradient ####################")
    a = tf.linspace(-1., 1, 10)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(a)
        y_1 = tf.nn.relu(a)
        y_2 = tf.nn.leaky_relu(a)

    # 求解a中每个点的梯度
    grad_1 = tape.gradient(y_1, [a])
    grad_2 = tape.gradient(y_2, [a])
    print('grad_1 = {}'.format(grad_1))
    print('grad_2 = {}'.format(grad_2))
    return None


if __name__ == '__main__':
    tf_sigmoid_gradient()
    tf_tanh_gradient()
    tf_relu_gradient()