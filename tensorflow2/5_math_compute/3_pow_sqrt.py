# -*- coding: utf-8 -*-
# @Time : 2019/12/28 0:46
# @Author : sxp
# @Email : 
# @File : 3_pow_sqrt.py
# @Project : python_common

import tensorflow as tf

def tf_pow_sqrt():
    """求矩阵中 对应元素 的平方和开方"""
    # t = tf.random.normal([2,3])
    t = tf.fill([2,2],4.)
    # 求三次方，下面两种方法相同
    t_1 = tf.pow(t,3)
    t_2 = t ** 3
    # 求开方
    t_3 = tf.sqrt(t)

    print('t_1 = {}'.format(t_1))
    print('t_2 = {}'.format(t_2))
    print('t_3 = {}'.format(t_3))#发现矩阵中元素必须为浮点型，否则报错

    # t_1 = [[8 8]
    #  [8 8]]

    # t_2 = [[8 8]
    #  [8 8]]

    # t_3 = [[2. 2.]
    #  [2. 2.]]
    return None



if __name__ == '__main__':
    tf_pow_sqrt()