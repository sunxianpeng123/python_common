# -*- coding: utf-8 -*-
# @Time : 2019/12/28 0:21
# @Author : sxp
# @Email : 
# @File : 2_log_exp.py
# @Project : python_common

import tensorflow as tf

def tf_math_log_exp():
    """tf.math.log：对矩阵中每个元素，求以e为底的对数，即自然对数
       tf.exp 和 tf.math.exp(x)：对矩阵中每个元素，求以e为底的x次方
     """
    print("#################1、自然对数和自然指数##################")
    t = tf.ones([2,2])
    t_1 = tf.math.log(t)
    t_2 = tf.math.exp(t)
    t_3 = tf.exp(t)

    print('t shape = {}'.format(t.shape))
    print('t_1 shape = {}'.format(t_1.shape))
    print('t_2 shape = {}'.format(t_2.shape))
    print('t_3 shape = {}'.format(t_3.shape))
    return None

def tf_log_ab():
    """求 类似 log 以 a 为底 b 的对数 """
    print("#################1、求以 a 为底 b 的对数##################")
    # 以 2 为底，8的对数
    a = tf.math.log(8.) / tf.math.log(2.)
    # 以 10 为底，100的对数
    b = tf.math.log(100.) / tf.math.log(10.)

    print('a = {}'.format(a))
    print('b = {}'.format(b))
    # a = 3.0
    # b = 2.0
    return None

if __name__ == '__main__':
    tf_math_log_exp()
    tf_log_ab()