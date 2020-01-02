# -*- coding: utf-8 -*-
# @Time : 2020/1/3 0:12
# @Author : sxp
# @Email : 
# @File : 8_output.py
# @Project : python_common

import tensorflow as tf

def tf_sigmoid():
    """ 0 - 1"""
    print("###################1、tf_sigmoid #########################")
    a = tf.linspace(-6., 6, 5)
    sigmoid = tf.sigmoid(a)

    print('a = {}'.format(a))
    # 将 a 中的每个实数 压缩在 0-1 之间
    print('sigmoid = {}'.format(sigmoid))
    # a = [-6. -3.  0.  3.  6.]
    # sigmoid = [0.00247262 0.04742587 0.5        0.95257413 0.9975274 ]
    print("====================")
    # 将图片的 单个通道 压缩在 0-1 之间
    x = tf.random.normal([1, 28, 28]) * 5
    print(tf.reduce_min(x))
    print(tf.reduce_max(x))
    # tf.Tensor(-13.3578205, shape=(), dtype=float32)
    # tf.Tensor(17.585007, shape=(), dtype=float32)

    x = tf.sigmoid(x)
    print(tf.reduce_min(x))
    print(tf.reduce_max(x))
    # tf.Tensor(0.0, shape=(), dtype=float32)
    # tf.Tensor(0.9999981, shape=(), dtype=float32)

    return None

def tf_softmax():
    print("###################2、tf_softmax #########################")
    a = tf.linspace(-2., 2, 5)
    softmax = tf.nn.softmax(a)

    print('a = {}'.format(a))
    # 将 a 压缩在 0-1 之间，并且压缩后的tensor个元素之和为1
    print('softmax = {}'.format(softmax))
    print('softmax sum = {}'.format(tf.reduce_sum(softmax)))
    # a = [-2. -1.  0.  1.  2.]
    # softmax = [0.01165623 0.03168492 0.08612854 0.23412165 0.6364086 ]
    # softmax sum = 1.0
    return None

def tf_softmax_classification():
    print("###################3、tf_softmax_classification #################")
    logits = tf.random.uniform([1, 5], minval=-2, maxval=2)
    preds = tf.nn.softmax(logits)

    print('logits = {}'.format(logits))
    print('preds = {}'.format(preds))
    print('logits sum = {}'.format(tf.reduce_sum(preds)))
    # logits = [[-1.8058162 -1.9695163  1.132618   0.9545622  1.5031219]]
    # preds = [[0.01565016 0.01328693 0.29557276 0.24736361 0.4281266 ]]
    # logits sum = 1.0000001192092896
    return None

def tf_tanh():
    print("###################4、tf_tanh #################")
    a = tf.linspace(-2., 2, 5)
    tanh = tf.nn.tanh(a)

    print('a = {}'.format(a))
    print('tanh = {}'.format(tanh))

    # a = [-2. -1.  0.  1.  2.]
    # tanh = [-0.9640276 -0.7615942  0.         0.7615942  0.9640276]
    return None

if __name__ == '__main__':
    tf_sigmoid()
    tf_softmax()
    tf_softmax_classification()
    tf_tanh()