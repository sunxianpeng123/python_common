# encoding: utf-8

"""
@author: sunxianpeng
@file: 2-segmentation.py
@time: 2019/12/28 19:53
"""

import tensorflow as tf



def tf_unstack():
    """分割"""
    print("##################1、tf_unstack#####################")

    a = tf.random.normal([4, 35, 8])
    b = tf.random.normal([4, 35, 8])

    t = tf.stack([a,b],axis=0) #t shape = (2, 4, 35, 8)
    # 在axis=0即第一个维度分开，分开后的数量是第一维度的shape大小，返回一个list
    # 分开后的tensor维度均相同
    t_1 = tf.unstack(t, axis=0)
    t_1_1 = t_1[0]
    t_1_2 = t_1[1]
    # 在axis=3即第4个维度分开，分开后的数量是第4维度的shape大小，返回一个list
    t_2 = tf.unstack(t,axis=3)

    print('t shape = {}'.format(t.shape))
    print('t_1 type = {},count = {}'.format(type(t_1),len(t_1)))
    print('t_1_1 shape = {}'.format(t_1_1.shape))
    print('t_1_2 shape = {}'.format(t_1_2.shape))
    print('t_2 type = {},count = {}'.format(type(t_2),len(t_2)))
    # t shape = (2, 4, 35, 8)
    # t_1 type = <class 'list'>, count = 2
    # t_1_1 shape = (4, 35, 8)
    # t_1_2 shape = (4, 35, 8)
    # t_2 type = <class 'list'>, count = 8
    return None


def tf_split():
    """分割"""
    print("##################2、tf_split#####################")
    a = tf.random.normal([4, 35, 8])
    b = tf.random.normal([4, 35, 8])

    t = tf.stack([a,b],axis=0) #t shape = (2, 4, 35, 8)

    # 在 axis=3(即第4个维度)分开，分开后的数量是 num_or_size_splits，并且 num_or_size_splits 必须可以整除axis指定维度处的 shape 大小
    t_1 = tf.split(t,axis=3,num_or_size_splits=2)
    t_1_1 = t_1[0]
    t_1_2 = t_1[1]

    print('t shape = {}'.format(t.shape))
    print('t_1 type = {},count = {}'.format(type(t_1),len(t_1)))
    print('t_1_1 shape = {}'.format(t_1_1.shape))
    print('t_1_2 shape = {}'.format(t_1_2.shape))

    # t shape = (2, 4, 35, 8)
    # t_1 type = <class 'list'>,count = 2
    # t_1_1 shape = (2, 4, 35, 4)
    # t_1_2 shape = (2, 4, 35, 4)
    print("===============")
    # 在 axis=3即第4个维度分开,分开后的每个tensor包含的数量大小根据num_or_size_splits指定得到
    # 并且num_or_size_splits中的  数字之和 必须和指定的axis处 shape 大小相同
    # 并且返回的顺序和 num_or_size_splits 的顺序相同
    t_2 = tf.split(t,axis=3,num_or_size_splits=[2,2,4])
    t_2_1 = t_2[0]
    t_2_2 = t_2[1]
    t_2_3 = t_2[2]
    print('t_2 type = {},count = {}'.format(type(t_2),len(t_2)))
    print('t_2_1 shape = {}'.format(t_2_1.shape))
    print('t_2_2 shape = {}'.format(t_2_2.shape))
    print('t_2_3 shape = {}'.format(t_2_3.shape))
    # t_2 type = <class 'list'>,count = 3
    # t_2_1 shape = (2, 4, 35, 2)
    # t_2_2 shape = (2, 4, 35, 2)
    # t_2_3 shape = (2, 4, 35, 4)
    return None


if __name__ == '__main__':
    tf_unstack()
    tf_split()