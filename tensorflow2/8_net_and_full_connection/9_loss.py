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
    print('**************求解熵**************')
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

    print('**************求解分类交叉熵**************')
    print("第一种方式求解交叉熵==============")
    y = tf.constant([0, 1, 0, 0])#分类标签
    preds_1 = tf.constant([0.25, 0.25, 0.25, 0.25])#预测概率
    preds_2 = tf.constant([0.1, 0.1, 0.8, 0.1])#预测概率
    preds_3 = tf.constant([0.1, 0.7, 0.1, 0.1])#预测概率
    preds_4 = tf.constant([0.01, 0.97, 0.01, 0.01])#预测概率

    cross_entropy_1 = tf.losses.categorical_crossentropy(y,preds_1)
    cross_entropy_2 = tf.losses.categorical_crossentropy(y,preds_2)
    cross_entropy_3 = tf.losses.categorical_crossentropy(y,preds_3)
    cross_entropy_4 = tf.losses.categorical_crossentropy(y,preds_4)

    print('cross_entropy_1 = {}'.format(cross_entropy_1))
    print('cross_entropy_2 = {}'.format(cross_entropy_2))
    print('cross_entropy_3 = {}'.format(cross_entropy_3))
    print('cross_entropy_4 = {}'.format(cross_entropy_4))
    # cross_entropy_1 = 1.3862943649291992
    # cross_entropy_2 = 2.397895336151123
    # cross_entropy_3 = 0.3566749691963196
    # cross_entropy_4 = 0.030459178611636162
    print("第二种方式求解交叉熵==============")
    multi_criteon = tf.losses.CategoricalCrossentropy()
    binary_criteon = tf.losses.BinaryCrossentropy()

    cross_entropy_5 = multi_criteon(y,preds_3)#与cross_entropy_3相同
    """二分类交叉熵"""
    cross_entropy_6 = multi_criteon([0, 1], [0.9, 0.1])
    cross_entropy_7 = binary_criteon([0, 1], [0.9, 0.1])
    cross_entropy_8 = tf.losses.binary_crossentropy([0, 1], [0.9, 0.1])

    print('cross_entropy_5 = {}'.format(cross_entropy_5))
    print('cross_entropy_6 = {}'.format(cross_entropy_6))
    print('cross_entropy_7 = {}'.format(cross_entropy_7))
    print('cross_entropy_8 = {}'.format(cross_entropy_8))

    return None

def tf_numberical_stability():
    """神经网络输出后，一般是先计算softmax，再计算交叉熵得到损失。但是此过程可能会出现计算问题（被0除），
    tensorflow 自带处理该问题的方法"""
    print("###################3、tf_numberical_stability #####################")
    x = tf.random.normal([1, 784])
    w = tf.random.normal([784, 2])
    b = tf.zeros([2])
    # logits指的是神经网络最后一层输出，并且未经激活函数激活
    logits = x @ w + b

    # 推荐这种方式
    cross_entropy_1 = tf.losses.categorical_crossentropy([0, 1], logits,from_logits=True)

    # 不推荐这种方式
    preds = tf.math.softmax(logits, axis=1)
    cross_entropy_2 = tf.losses.categorical_crossentropy([0, 1],preds)
    print('cross_entropy_1 = {}'.format(cross_entropy_1))
    print('cross_entropy_2 = {}'.format(cross_entropy_2))

    return None


if __name__ == '__main__':
    tf_mse_norm_loss()
    tf_cross_entropy()
    tf_numberical_stability()