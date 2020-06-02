# encoding: utf-8

"""
@author: sunxianpeng
@file: 5_argmax_argmin.py
@time: 2019/12/29 20:42
"""

import tensorflow as tf

def tf_argmax():
    """求解tensor中最大值的索引 """
    print("##################1、tf_argmax#####################")
    t = tf.random.normal([2,3])
    # 下面两组解相同
    # axis=0 表示以第一个维度为标准，求解每组数据(每组数据两个值，即求每列)中的最大值索引
    t_1 = tf.argmax(t) #axis 默认为 0
    t_2 = tf.argmax(t, axis=0)

    t_3 = tf.argmax(t, axis=1)
    print (t.shape)
    print('t = {}'.format(t))
    print('t_1 = {}'.format(t_1))
    print('t_2 = {}'.format(t_2))
    print('t_3 = {}'.format(t_3))
    # t = [[ 0.32165205  0.41186765 -1.4489236 ]
    #       [ 0.98179  0.13301654  1.3752135 ]]
    # t_1 = [1 0 1]
    # t_2 = [1 0 1]
    # t_3 = [1 2]
    return None

def tf_argmin():
    """求解tensor中最小值的索引 """
    print("##################2、tf_argmin#####################")
    t = tf.random.normal([2,3])

    # 下面两组解相同
    # axis=0 表示以第一个维度为标准，求解每组数据(每组数据两个值，即求每列)中的最小值索引
    t_1 = tf.argmin(t)
    t_2 = tf.argmin(t, axis=0)
    t_3 = tf.argmin(t, axis=1)

    print('t = {}'.format(t))
    print('t_1 = {}'.format(t_1))
    print('t_2 = {}'.format(t_2))
    print('t_3 = {}'.format(t_3))
    # t = [[-0.9748641  -0.8209598  -0.02690448]
    #  [ 0.69590753 -1.0007102  -0.44212893]]

    # t_1 = [0 1 1]
    # t_2 = [0 1 1]
    # t_3 = [0 1]
    return None

if __name__ == '__main__':
    tf_argmax()
    tf_argmin()