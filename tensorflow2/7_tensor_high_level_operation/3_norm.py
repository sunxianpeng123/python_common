# encoding: utf-8

"""
@author: sunxianpeng
@file: 3_norm.py
@time: 2019/12/29 19:52
"""
"""范数"""

import tensorflow as tf


def tf_l2_norm():
    """ L2 范数"""
    print("################1、tf_l2_norm ################")
    print("求解整个tensor的L2范数============")
    # 以下两种求l2范数的结果时相同的
    a = tf.ones([2,2])
    a_1 = tf.norm(a)
    a_2 = tf.sqrt(tf.reduce_sum(tf.square(a)))

    b = tf.ones([4,28,28,3])
    # 以下两种求l2范数的结果时相同的
    b_1 = tf.norm(b)
    b_2 = tf.sqrt(tf.reduce_sum(tf.square(b)))

    print('a_1 = {}'.format(a_1))
    print('a_2 = {}'.format(a_2))
    print('b_1 = {}'.format(b_1))
    print('b_2 = {}'.format(b_2))
    # a_1 = 2.0
    # a_2 = 2.0
    # b_1 = 96.99484252929688
    # b_2 = 96.99484252929688
    print("指定tensor的维度，求解L2范数============")
    c = tf.ones([2,2])

    # 求解整个tensor的L2范数
    c_1 = tf.norm(c)
    # 指定axis=1 即按照第二个维度，求解范数，
    # 此处即以行为单位，求解每行的L2范数
    c_2 = tf.norm(c,ord=2,axis=1)

    print('c_1 = {}'.format(c_1))
    print('c_2 = {}'.format(c_2))
    # c_1 = 2.0
    # c_2 = [1.4142135 1.4142135]

    return None

def tf_l1_norm():
    """ L1 范数"""
    print("################2、tf_l1_norm ################")
    print("求解整个tensor的L1范数============")
    # 以下两种求l2范数的结果时相同的
    a = tf.ones([2,2])
    a_1 = tf.norm(a,ord=1)#L1 范数

    print('a_1 = {}'.format(a_1))
    # a_1 = 4.0
    print("指定tensor的维度，求解L1范数============")
    # 指定axis=0 即按照第一个维度，求解范数，
    # 此处即以列为单位，求解每列的L1范数
    a_2 = tf.norm(a,ord=1,axis=0)
    # 指定axis=1 即按照第二个维度，求解范数，
    # 此处即以行为单位，求解每行的L1范数
    a_3 = tf.norm(a,ord=1,axis=1)

    print('a_2 = {}'.format(a_2))
    print('a_3 = {}'.format(a_3))
    # a_2 = [2. 2.]
    # a_3 = [2. 2.]
    return None

if __name__ == '__main__':
    tf_l2_norm()
    tf_l1_norm()