# -*- coding: utf-8 -*-
# @Time : 2019/12/31 0:22
# @Author : sxp
# @Email : 
# @File : 9_set_tensor_value_range_1.py
# @Project : python_common

import tensorflow as tf

def tf_clip_by_value():
    """设置张量中值得取值范围"""
    print("##############1、tf.maximum  minimum  clip_by_value#################")
    t = tf.random.uniform([3,3],maxval=10,minval=0,dtype=tf.int32)
    # 实现的max(k,x)函数，如果x<k，则x设置为k，否则不变，此处tensor中值小于3则设置为3，否则不变
    t_1 = tf.maximum(t, 3)
    # 实现的min(k,x)函数，如果x>k，则x设置为k，否则不变，此处tensor中值大于4则设置为4，否则不变
    t_2 = tf.minimum(t, 4)
    # 实现将tensor中的值范围设置一个范围,下面三种方式结果相同
    t_3 = tf.clip_by_value(t, 2, 5)#小于2 设置变为2 ，大于5变为 5
    t_4 = tf.maximum(tf.minimum(t,5),2)
    t_5 = tf.minimum(tf.maximum(t,2),5)

    print('t = {}'.format(t))
    print('t_1 = {}'.format(t_1))
    print('t_2 = {}'.format(t_2))
    print('t_3 = {}'.format(t_3))
    print('t_4 = {}'.format(t_4))
    print('t_5 = {}'.format(t_5))
    # t = [[6 4 5]
    #  [4 6 2]
    #  [4 7 4]]

    # t_1 = [[6 4 5]
    #  [4 6 3]
    #  [4 7 4]]

    # t_2 = [[4 4 4]
    #  [4 4 2]
    #  [4 4 4]]

    # t_3 = [[5 4 5]
    #  [4 5 2]
    #  [4 5 4]]

    # t_4 = [[5 4 5]
    #  [4 5 2]
    #  [4 5 4]]

    # t_5 = [[5 4 5]
    #  [4 5 2]
    #  [4 5 4]]
    return None

def tf_relu():
    """ """
    print("##############2、tf_relu #################")
    t = tf.random.uniform([3, 3],minval=0,maxval=10,dtype=tf.int32)
    t = t - 5
    # 实现的max(0,x)函数，如果x<0，则x设置为0，否则不变，
    t_1 = tf.nn.relu(t)
    t_2 = tf.maximum(t, 0)

    print('t = {}'.format(t))
    print('t_1 = {}'.format(t_1))
    print('t_2 = {}'.format(t_2))
    # t = [[-3  3 -3]
    #  [-2  1  3]
    #  [-2 -3 -4]]

    # t_1 = [[0 3 0]
    #  [0 1 3]
    #  [0 0 0]]

    # t_2 = [[0 3 0]
    #  [0 1 3]
    #  [0 0 0]]
    return None

if __name__ == '__main__':
    tf_clip_by_value()
    tf_relu()