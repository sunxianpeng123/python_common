# -*- coding: utf-8 -*-
# @Time : 2020/1/3 0:39
# @Author : sxp
# @Email : 
# @File : 9_loss.py
# @Project : python_common

import tensorflow as tf

def tf_mse_norm_loss():
    """均方误差"""
    print("###################1、tf_mse_norm_loss #####################")
    # 分类标签
    y = tf.constant([1, 2, 3, 0, 2])
    # 独热编码，每个分类标签变成一个向量
    y = tf.one_hot(y, depth=4)
    y = tf.cast(y, dtype=tf.float32)
    print('y = {}'.format(y))
    # y = [[0. 1. 0. 0.]
    #  [0. 0. 1. 0.]
    #  [0. 0. 0. 1.]
    #  [1. 0. 0. 0.]
    #  [0. 0. 1. 0.]]

    # 模拟神经网络的输出
    preds = tf.random.normal([5, 4])
    # 求mse,下面两种方式相同
    loss_mse_1 = tf.reduce_mean(tf.square(y - preds))
    loss_mse_2 = tf.reduce_mean(tf.losses.MSE(y, preds))
    print('loss_mse_1 = {}'.format(loss_mse_1))
    print('loss_mse_2 = {}'.format(loss_mse_2))
    # loss_mse_1 = 1.2089463472366333
    # loss_mse_2 = 1.2089463472366333
    print("=========================")

    loss_norm_1 = tf.norm(y - preds)
    print('loss_norm_1 = {}'.format(loss_norm_1))
    # loss_norm_1 = 5.651540756225586
    return None

def tf_cross_entropy():
    """交叉熵,越大越稳定，惊喜度越小，信息越小"""
    print("###################2、tf_cross_entropy #####################")
    print('求解熵==============')
    a = tf.fill([4], 0.25)
    entropy_1 = - tf.reduce_sum(a * tf.math.log(a) / tf.math.log(2.))
    print('entropy_1 = {}'.format(entropy_1))
    # entropy_1 = 2.0
    b = tf.constant([0.1, 0.1, 0.1, 0.7])
    entropy_2 = - tf.reduce_sum(b * tf.math.log(b) / tf.math.log(2.))
    print('entropy_2 = {}'.format(entropy_2))
    # entropy_2 = 1.35677969455719
    c = tf.constant([0.01, 0.01, 0.01, 0.97])
    entropy_3 = - tf.reduce_sum(c * tf.math.log(c) / tf.math.log(2.))
    print('entropy_3 = {}'.format(entropy_3))
    # entropy_3 = 0.2419406771659851

    
    return None

if __name__ == '__main__':
    tf_mse_norm_loss()
    tf_cross_entropy()