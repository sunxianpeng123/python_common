# -*- coding: utf-8 -*-
# @Time : 2019/12/26 0:46
# @Author : sxp
# @Email : 
# @File : 3_increase_and_reduce_dims.py
# @Project : python_common

import tensorflow as tf

def tf_expand_dims():
    """增加维度"""
    print("###############1、tf_expand_dims增加维度################")
    t = tf.random.normal([4,35,8])#4个班级，35个学生，8门课程
    t_1 = tf.expand_dims(t,axis=0)#axis指定在t的哪个维度前增加一个维度,学校
    # t有三个维度，则有两个空隙，加上两头共四个增加维度的位置（下标即为0,1,2,3），3表示在最后加一个维度
    t_2 = tf.expand_dims(t,axis=3)
    t_3 = tf.expand_dims(t,axis=-1)#axis=-1表示在最后增加一个维度
    # t有三个维度，则有两个空隙，加上两头共四个增加维度的位置，-4表示在开头增加维度
    t_4 = tf.expand_dims(t,axis=-4)

    print('t shape = {}'.format(t.shape))
    print('t_1 shape = {}'.format(t_1.shape))
    print('t_2 shape = {}'.format(t_2.shape))
    print('t_3 shape = {}'.format(t_3.shape))
    print('t_4 shape = {}'.format(t_4.shape))
    # t shape = (4, 35, 8)
    # t_1 shape = (1, 4, 35, 8)
    # t_2 shape = (4, 35, 8, 1)
    # t_3 shape = (4, 35, 8, 1)
    # t_4 shape = (1, 4, 35, 8)
    return None

def tf_squeeze_dims():
    """减少维度，不能减少维度不为 1 的维度"""
    print("###############2、tf_squeeze_dims减少维度################")
    t = tf.zeros([1,2,1,3])
    t_1 = tf.squeeze(t)#将t中维度为1去的全部去掉
    t_2 = tf.squeeze(t,axis=0)#去掉t中下标0处的维度
    t_3 = tf.squeeze(t,axis=-2)#去掉t中下标2（倒数第2个）处的维度
    t_4 = tf.squeeze(t,axis=-4)#去掉t中下标0（倒数第4个）处的维度

    print('t shape = {}'.format(t.shape))
    print('t_1 shape = {}'.format(t_1.shape))
    print('t_2 shape = {}'.format(t_2.shape))
    print('t_3 shape = {}'.format(t_3.shape))
    print('t_4 shape = {}'.format(t_4.shape))
    # t shape = (1, 2, 1, 3)
    # t_1 shape = (2, 3)
    # t_2 shape = (2, 1, 3)
    # t_3 shape = (1, 2, 3)
    # t_4 shape = (2, 1, 3)
    return None

if __name__ == '__main__':
    tf_expand_dims()
    tf_squeeze_dims()