# -*- coding: utf-8 -*-
# @Time : 2019/12/26 0:26
# @Author : sxp
# @Email : 
# @File : 2_transpose.py
# @Project : python_common

import tensorflow as tf

def tf_transpose():
    """ tensor的维度交换（转置）"""
    print("################1、tf_transpose######################")
    t = tf.random.normal((4,3,2,1))
    # 将t的维度信息倒置
    t_1 = tf.transpose(t)
    # 指定t_2的维度顺序，即perm指的是t中维度的下标,按照下标顺序指定t的shape
    t_2 = tf.transpose(t,perm=[0,1,3,2])

    print('t shape ={}'.format(t.shape))
    print('t_1 shape = {}'.format(t_1.shape))
    print('t_2 shape = {}'.format(t_2.shape))
    # t shape =(4, 3, 2, 1)
    # t_1 shape =(1, 2, 3, 4)
    # t_2 shape =(4, 3, 1, 2)
    return None


if __name__ == '__main__':
    tf_transpose()