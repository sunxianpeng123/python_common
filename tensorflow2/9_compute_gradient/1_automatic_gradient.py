# encoding: utf-8

"""
@author: sunxianpeng
@file: 1_automatic_gradient.py
@time: 2020/1/4 14:52
"""

import tensorflow as tf

def tf_gradient_tape():
    """ 一阶求导"""
    print("#################1、tf_gradient_tape ###############")
    print("对同一个变量只求解一次梯度==============")
    w = tf.constant([1.])
    x = tf.constant([2.])
    y = x * w
    with tf.GradientTape() as tape:
        tape.watch([w])
        y2 = x * w
    grad_2 = tape.gradient(y2, [w])

    print('grad_2 = {}'.format(grad_2))
    # grad1 = [<tf.Tensor: id=8, shape=(1,), dtype=float32, numpy=array([2.], dtype=float32)>]

    print("对同一个变量求解多次梯度==============")
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([w])
        y3 = 2 * x * w
    grad_3 = tape.gradient(y3, [w])
    grad_4 = tape.gradient(y3, [w])
    print('grad_3 = {}'.format(grad_3))
    print('grad_4 = {}'.format(grad_4))
    # grad_3 = [<tf.Tensor: id=14, shape=(1,), dtype=float32, numpy=array([2.], dtype=float32)>]
    # grad_4 = [<tf.Tensor: id=19, shape=(1,), dtype=float32, numpy=array([2.], dtype=float32)>]
    return None

def tf_2nd_order_gradient_tape():
    """二阶求导
    y = xw + b,
    y关于w的一阶导数:gradient_w_1 = dy/dw = x
    y关于w的二阶导数:gradient_w_2 = d(gradient_w_1)/dw = None
    """
    print("#################2、tf_2nd_order_gradient_tape ###############")
    w = tf.Variable(1.0)
    b = tf.Variable(2.0)
    x = tf.Variable(3.0)

    with tf.GradientTape() as t1:
        with tf.GradientTape() as t2:
            y = x * w + b
        gradient_w_1 = t2.gradient(y, [w, b])
    gradient_w_2 = t1.gradient(gradient_w_1, w)
    print('gradient_w_1 = {}'.format(gradient_w_1))
    print('gradient_w_2 = {}'.format(gradient_w_2))
    # gradient_w_1 = [<tf.Tensor: id=49, shape=(), dtype=float32, numpy=3.0>, <tf.Tensor: id=47, shape=(), dtype=float32, numpy=1.0>]
    # gradient_w_2 = None
    return None

if __name__ == '__main__':
    tf_gradient_tape()
    tf_2nd_order_gradient_tape()